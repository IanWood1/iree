// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCBuilders.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir::iree_compiler::emitc_builders {

namespace {
std::string mapPreprocessorDirective(PreprocessorDirective directive) {
  switch (directive) {
  case PreprocessorDirective::DEFINE:
    return "#define";
  case PreprocessorDirective::UNDEF:
    return "#undef";
  case PreprocessorDirective::IFDEF:
    return "#ifdef";
  case PreprocessorDirective::IFNDEF:
    return "#ifndef";
  case PreprocessorDirective::IF:
    return "#if";
  case PreprocessorDirective::ENDIF:
    return "#endif";
  case PreprocessorDirective::ELSE:
    return "#else";
  case PreprocessorDirective::ELIF:
    return "#elif";
  case PreprocessorDirective::LINE:
    return "#line";
  case PreprocessorDirective::ERROR:
    return "#error";
  case PreprocessorDirective::INCLUDE:
    return "#include";
  case PreprocessorDirective::PRAGMA:
    return "#pragma";
  default:
    llvm_unreachable("unsupported unary operator");
    return "XXX";
  }
}
} // namespace

Value allocateVariable(OpBuilder builder, Location location, Type type,
                       Attribute initializer) {
  return builder
      .create<emitc::VariableOp>(
          /*location=*/location,
          /*resultType=*/type,
          /*value=*/initializer)
      .getResult();
}

Value allocateVariable(OpBuilder builder, Location location, Type type,
                       std::optional<StringRef> initializer) {
  auto ctx = builder.getContext();
  return allocateVariable(
      builder, location, type,
      emitc::OpaqueAttr::get(ctx, initializer.value_or("")));
}

Value addressOf(OpBuilder builder, Location location, Value operand) {
  auto ctx = builder.getContext();

  return builder
      .create<emitc::ApplyOp>(
          /*location=*/location,
          /*result=*/emitc::PointerType::get(operand.getType()),
          /*applicableOperator=*/StringAttr::get(ctx, "&"),
          /*operand=*/operand)
      .getResult();
}

Value contentsOf(OpBuilder builder, Location location, Value operand) {
  auto ctx = builder.getContext();

  Type type = operand.getType();
  assert(isa<emitc::PointerType>(type));

  return builder
      .create<emitc::ApplyOp>(
          /*location=*/location,
          /*result=*/llvm::cast<emitc::PointerType>(type).getPointee(),
          /*applicableOperator=*/StringAttr::get(ctx, "*"),
          /*operand=*/operand)
      .getResult();
}

Value sizeOf(OpBuilder builder, Location loc, Value value) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOpaqueOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_host_size_t"),
          /*callee=*/"sizeof",
          /*operands=*/ArrayRef<Value>{value})
      .getResult(0);
}

Value sizeOf(OpBuilder builder, Location loc, Attribute attr) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOpaqueOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_host_size_t"),
          /*callee=*/"sizeof",
          /*operands=*/ArrayRef<Value>{},
          /*args=*/ArrayAttr::get(ctx, {attr}))
      .getResult(0);
}

void memcpy(OpBuilder builder, Location location, Value dest, Value src,
            Value count) {
  builder.create<emitc::CallOpaqueOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/"memcpy",
      /*operands=*/ArrayRef<Value>{dest, src, count});
}

void memset(OpBuilder builder, Location location, Value dest, int ch,
            Value count) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOpaqueOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/"memset",
      /*operands=*/ArrayRef<Value>{dest, count},
      /*args=*/
      ArrayAttr::get(ctx,
                     {builder.getIndexAttr(0), builder.getUI32IntegerAttr(ch),
                      builder.getIndexAttr(1)}));
}

Value arrayElement(OpBuilder builder, Location location, Type type,
                   size_t index, Value operand) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOpaqueOp>(
          /*location=*/location,
          /*type=*/type,
          /*callee=*/"EMITC_ARRAY_ELEMENT",
          /*operands=*/ArrayRef<Value>{operand},
          /*args=*/
          ArrayAttr::get(
              ctx, {builder.getIndexAttr(0), builder.getI32IntegerAttr(index)}))
      .getResult(0);
}

Value arrayElementAddress(OpBuilder builder, Location location, Type type,
                          IntegerAttr index, Value operand) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOpaqueOp>(
          /*location=*/location,
          /*type=*/type,
          /*callee=*/"EMITC_ARRAY_ELEMENT_ADDRESS",
          /*operands=*/ArrayRef<Value>{operand},
          /*args=*/
          ArrayAttr::get(ctx, {builder.getIndexAttr(0), index}))
      .getResult(0);
}

Value arrayElementAddress(OpBuilder builder, Location location, Type type,
                          Value index, Value operand) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOpaqueOp>(
          /*location=*/location,
          /*type=*/type,
          /*callee=*/"EMITC_ARRAY_ELEMENT_ADDRESS",
          /*operands=*/ArrayRef<Value>{operand, index},
          /*args=*/
          ArrayAttr::get(ctx,
                         {builder.getIndexAttr(0), builder.getIndexAttr(1)}),
          /*templateArgs=*/ArrayAttr{})
      .getResult(0);
}

void arrayElementAssign(OpBuilder builder, Location location, Value array,
                        size_t index, Value value) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOpaqueOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/"EMITC_ARRAY_ELEMENT_ASSIGN",
      /*operands=*/ArrayRef<Value>{array, value},
      /*args=*/
      ArrayAttr::get(ctx,
                     {builder.getIndexAttr(0), builder.getI32IntegerAttr(index),
                      builder.getIndexAttr(1)}));
}

void structDefinition(OpBuilder builder, Location location,
                      StringRef structName, ArrayRef<StructField> fields) {
  auto ctx = builder.getContext();
  std::string decl = std::string("struct ") + structName.str() + " {";
  for (auto &field : fields) {
    decl += field.type + " " + field.name;
    if (field.isArray())
      decl += "[" + std::to_string(field.arraySize.value()) + "]";
    decl += ";";
  }
  decl += "};";

  builder.create<emitc::VerbatimOp>(location, StringAttr::get(ctx, decl));
}

Value structMember(OpBuilder builder, Location location, Type type,
                   StringRef memberName, Value operand) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOpaqueOp>(
          /*location=*/location,
          /*type=*/type,
          /*callee=*/"EMITC_STRUCT_MEMBER",
          /*operands=*/ArrayRef<Value>{operand},
          /*args=*/
          ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                               emitc::OpaqueAttr::get(ctx, memberName)}))
      .getResult(0);
}

void structMemberAssign(OpBuilder builder, Location location,
                        StringRef memberName, Value operand, Value data) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOpaqueOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/"EMITC_STRUCT_MEMBER_ASSIGN",
      /*operands=*/ArrayRef<Value>{operand, data},
      /*args=*/
      ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                           emitc::OpaqueAttr::get(ctx, memberName),
                           builder.getIndexAttr(1)}));
}

void structMemberAssign(OpBuilder builder, Location location,
                        StringRef memberName, Value operand, StringRef data) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOpaqueOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/"EMITC_STRUCT_MEMBER_ASSIGN",
      /*operands=*/ArrayRef<Value>{operand},
      /*args=*/
      ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                           emitc::OpaqueAttr::get(ctx, memberName),
                           emitc::OpaqueAttr::get(ctx, data)}));
}

Value structPtrMember(OpBuilder builder, Location location, Type type,
                      StringRef memberName, Value operand) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOpaqueOp>(
          /*location=*/location,
          /*type=*/type,
          /*callee=*/"EMITC_STRUCT_PTR_MEMBER",
          /*operands=*/ArrayRef<Value>{operand},
          /*args=*/
          ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                               emitc::OpaqueAttr::get(ctx, memberName)}))
      .getResult(0);
}

void structPtrMemberAssign(OpBuilder builder, Location location,
                           StringRef memberName, Value operand, Value data) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOpaqueOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/"EMITC_STRUCT_PTR_MEMBER_ASSIGN",
      /*operands=*/ArrayRef<Value>{operand, data},
      /*args=*/
      ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                           emitc::OpaqueAttr::get(ctx, memberName),
                           builder.getIndexAttr(1)}));
}

Value ireeMakeCstringView(OpBuilder builder, Location location,
                          std::string str) {
  std::string escapedStr;
  llvm::raw_string_ostream os(escapedStr);
  os.write_escaped(str);
  auto quotedStr = std::string("\"") + escapedStr + std::string("\"");

  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOpaqueOp>(
          /*location=*/location,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_string_view_t"),
          /*callee=*/"iree_make_cstring_view",
          /*operands=*/ArrayRef<Value>{},
          /*args=*/
          ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(ctx, quotedStr)}))
      .getResult(0);
}

Value ireeOkStatus(OpBuilder builder, Location location) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOpaqueOp>(
          /*location=*/location,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
          /*callee=*/"iree_ok_status",
          /*operands=*/ArrayRef<Value>{})
      .getResult(0);
}

Value ireeVmInstanceLookupType(OpBuilder builder, Location location,
                               Value instance, Value stringView) {
  auto ctx = builder.getContext();
  Type refType = emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t");
  return builder
      .create<emitc::CallOpaqueOp>(
          /*location=*/location,
          /*type=*/refType,
          /*callee=*/"iree_vm_instance_lookup_type",
          /*operands=*/ArrayRef<Value>{instance, stringView})
      .getResult(0);
}

void ireeVmRefRelease(OpBuilder builder, Location location, Value operand) {
  builder.create<emitc::CallOpaqueOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/"iree_vm_ref_release",
      /*operands=*/ArrayRef<Value>{operand});
}

emitc::VerbatimOp preprocessorDirective(OpBuilder builder, Location location,
                                        PreprocessorDirective directive,
                                        StringRef value) {

  auto t = mapPreprocessorDirective(directive);
  if (!value.empty()) {
    t += " ";
    t += value;
  }

  return builder.create<emitc::VerbatimOp>(location, t);
}

} // namespace mlir::iree_compiler::emitc_builders
