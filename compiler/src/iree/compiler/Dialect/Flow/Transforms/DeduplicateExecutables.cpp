// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mutex>
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Utils/EquivalenceUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-flow-deduplicate-executables"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_DEDUPLICATEEXECUTABLESPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

// Utilities to make SymbolRefAttr easier to construct.
static SymbolRefAttr nestSymbolRef(SymbolRefAttr baseRefAttr,
                                   FlatSymbolRefAttr leafRefAttr) {
  if (!baseRefAttr)
    return leafRefAttr;
  SmallVector<FlatSymbolRefAttr> nestedRefAttrs;
  llvm::append_range(nestedRefAttrs, baseRefAttr.getNestedReferences());
  nestedRefAttrs.push_back(leafRefAttr);
  return SymbolRefAttr::get(baseRefAttr.getContext(),
                            baseRefAttr.getRootReference(), nestedRefAttrs);
}
static SymbolRefAttr nestSymbolRef(SymbolRefAttr baseRefAttr,
                                   SymbolOpInterface leafOp) {
  return nestSymbolRef(baseRefAttr, FlatSymbolRefAttr::get(leafOp));
}

// Recursively gathers symbol->symbol replacements from the old object table
// regions to the new object table regions into |symbolReplacements|.
static void gatherReplacements(
    SymbolRefAttr oldSymbolRefAttr, MutableArrayRef<Region> oldRegions,
    SymbolRefAttr newSymbolRefAttr, MutableArrayRef<Region> newRegions,
    DenseMap<Attribute, SymbolRefAttr> &symbolReplacements) {
  for (auto [nestedOldRegion, nestedNewRegion] :
       llvm::zip_equal(oldRegions, newRegions)) {
    for (auto [oldNestedSymbolOp, newNestedSymbolOp] :
         llvm::zip_equal(nestedOldRegion.getOps<SymbolOpInterface>(),
                         nestedNewRegion.getOps<SymbolOpInterface>())) {
      if (!oldNestedSymbolOp.isPublic())
        continue; // ignore private symbols
      auto oldNestedSymbolRefAttr =
          nestSymbolRef(oldSymbolRefAttr, oldNestedSymbolOp);
      auto newNestedSymbolRefAttr =
          nestSymbolRef(newSymbolRefAttr, newNestedSymbolOp);
      symbolReplacements[oldNestedSymbolRefAttr] = newNestedSymbolRefAttr;
      gatherReplacements(oldNestedSymbolRefAttr,
                         oldNestedSymbolOp->getRegions(),
                         newNestedSymbolRefAttr,
                         newNestedSymbolOp->getRegions(), symbolReplacements);
    }
  }
}

// Replaces each usage of an entry point with its original symbol name with a
// new symbol name.
static void
replaceSymbolRefs(Operation *scopeOp,
                  const DenseMap<Attribute, SymbolRefAttr> &replacements) {
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](SymbolRefAttr oldAttr) {
    auto it = replacements.find(oldAttr);
    return std::make_pair(it == replacements.end() ? oldAttr : it->second,
                          WalkResult::skip());
  });
  for (auto &region : scopeOp->getRegions()) {
    for (auto funcOp : region.getOps<mlir::FunctionOpInterface>()) {
      funcOp->walk([&](Operation *op) {
        replacer.replaceElementsIn(op);
        return WalkResult::advance();
      });
    }
  }
}

// Returns the total number of objects deduplicated, if any.
// The provided |objects| array may have dead ops upon return.
static int deduplicateObjects(Operation *scopeOp,
                              ArrayRef<Operation *> allObjectOps) {
  // Bucket based on the hash of the names of at most the first 5 ops.
  // 5 was randomly chosen to be small enough to not increase overhead much,
  // but giving at least enough of a sample that there is some bucketing. This
  // was not empirically determined.
  LLVM_DEBUG({ llvm::dbgs() << "Start Pass!\n"; });
  llvm::MapVector<uint32_t, SmallVector<SymbolOpInterface>> objectMap;
  std::mutex mutex;
  constexpr int kMaxHashedOps = 5;
  mlir::parallelForEach(
      scopeOp->getContext(), allObjectOps, [&](auto objectOp) {
        int count = 0;
        llvm::hash_code hash(1);
        objectOp->walk([&](Operation *op) {
          hash = llvm::hash_combine(hash, op->getName());
          hash = llvm::hash_combine(hash, op->getResultTypes());
          return (++count >= kMaxHashedOps) ? WalkResult::interrupt()
                                            : WalkResult::advance();
        });
        std::lock_guard<std::mutex> lock(mutex);
        objectMap[hash_value(hash)].push_back(
            cast<SymbolOpInterface>(objectOp));
      });

  LLVM_DEBUG({
    llvm::dbgs() << "Finish Hash!\n";

    // Print the number of entries in the object map as well as the
    // average/min/max size.
    int totalObjects = 0;
    int minBucketSize = std::numeric_limits<int>::max();
    int maxBucketSize = 0;
    for (auto &entry : objectMap) {
      totalObjects += entry.second.size();
      minBucketSize = std::min(minBucketSize, (int)entry.second.size());
      maxBucketSize = std::max(maxBucketSize, (int)entry.second.size());
    }

    llvm::dbgs() << "Total objects: " << totalObjects << "\n";
    llvm::dbgs() << "Number of buckets: " << objectMap.size() << "\n";
    llvm::dbgs() << "Average bucket size: " << (totalObjects / objectMap.size())
                 << "\n";
    llvm::dbgs() << "Min bucket size: " << minBucketSize << "\n";
    llvm::dbgs() << "Max bucket size: " << maxBucketSize << "\n";
  });

  // For each object find the first object which it is equivalent to and record
  // the replacement.
  SetVector<Operation *> deadOps;
  DenseMap<Attribute, SymbolRefAttr> symbolReplacements;
  mlir::parallelForEach(scopeOp->getContext(), objectMap, [&](auto object) {
    auto &[key, objectOps] = object;
    (void)key;
    OperationEquivalenceCache equivalenceCache(scopeOp->getContext());
    for (int i = objectOps.size() - 1; i >= 0; --i) {
      auto duplicateOp = objectOps[i];
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (deadOps.contains(duplicateOp))
          continue;
      }
      for (int j = 0; j < i; ++j) {
        auto referenceOp = objectOps[j];
        {
          std::lock_guard<std::mutex> lock(mutex);
          if (deadOps.contains(referenceOp))
            continue;
        }

        // Compare this potentially duplicate object to the reference one.
        if (!isStructurallyEquivalentTo(equivalenceCache, *duplicateOp,
                                        *referenceOp)) {
          continue;
        }
        {
          std::lock_guard<std::mutex> lock(mutex);

          // Found an equivalent object! Record it and move on to the next.
          deadOps.insert(duplicateOp);

          // Record symbol reference replacements within nested objects.
          gatherReplacements(SymbolRefAttr::get(duplicateOp),
                             duplicateOp->getRegions(),
                             SymbolRefAttr::get(referenceOp),
                             referenceOp->getRegions(), symbolReplacements);
        }
        break;
      }
    }
  });

  LLVM_DEBUG({
    llvm::dbgs() << "Found duplicates!\n";
    llvm::dbgs() << "Number of dead ops: " << deadOps.size() << "\n";
  });

  // Replace all symbol references within the scope.
  replaceSymbolRefs(scopeOp, symbolReplacements);

  // Remove the duplicate objects now that they are no longer referenced.
  // We could rely on SymbolDCE for this but that makes looking at IR dumps
  // harder as after this pass runs and until SymbolDCE runs there are lots of
  // dead objects in the output.
  for (auto *op : deadOps)
    op->erase();

  return deadOps.size();
}

} // namespace

class DeduplicateExecutablesPass
    : public IREE::Flow::impl::DeduplicateExecutablesPassBase<
          DeduplicateExecutablesPass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    SmallVector<Operation *> allObjects;
    for (auto &op : moduleOp.getOps()) {
      if (op.hasTrait<OpTrait::IREE::Util::ObjectLike>())
        allObjects.push_back(&op);
    }
    if (allObjects.empty())
      return;
    (void)deduplicateObjects(moduleOp, allObjects);
    // totalObjects = allObjects.size();
    // objectsDeduplicated = deduplicateObjects(moduleOp, allObjects);
    // remainingObjects = totalObjects - objectsDeduplicated;
  }

private:
  // Statistic totalObjects{
  //     this,
  //     "total object(s)",
  //     "Number of object ops before deduplication",
  // };
  // Statistic objectsDeduplicated{
  //     this,
  //     "duplicate object(s)",
  //     "Number of object ops removed as duplicates",
  // };
  // Statistic remainingObjects{
  //     this,
  //     "unique object(s)",
  //     "Number of object ops remaining after deduplication",
  // };
};

} // namespace mlir::iree_compiler::IREE::Flow
