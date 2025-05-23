# TODO: (#17751) Add the arm_64 tests when the bug resolved. See:
#   https://github.com/iree-org/iree/actions/runs/10468944505/job/28990909321#step:4:9815
if(IREE_ARCH STREQUAL "arm_64")
  return()
endif()

iree_generated_e2e_runner_test(
  NAME
    e2e_attention_cpu_f16_f16_f16_small
  TEST_TYPE
    attention
  GENERATOR
    "generate_e2e_attention_tests.py"
  GENERATOR_ARGS
    "--query_type=f16"
    "--key_type=f16"
    "--value_type=f16"
    "--shapes=small"
  TEST_RUNNER
    iree_tools_testing_e2e_iree-e2e-attention-test
  TARGET_BACKENDS
    "llvm-cpu"
  DRIVERS
    "local-task"
  LABELS
    "hostonly"
    "local"
)

iree_generated_e2e_runner_test(
  NAME
    e2e_attention_cpu_f16_f16_f16_medium
  TEST_TYPE
    attention
  GENERATOR
    "generate_e2e_attention_tests.py"
  GENERATOR_ARGS
    "--query_type=f16"
    "--key_type=f16"
    "--value_type=f16"
    "--shapes=medium"
  TEST_RUNNER
    iree_tools_testing_e2e_iree-e2e-attention-test
  TARGET_BACKENDS
    "llvm-cpu"
  DRIVERS
    "local-task"
  LABELS
    "hostonly"
    "local"
)

iree_generated_e2e_runner_test(
  NAME
    e2e_attention_cpu_f16_f16_f16_large
  TEST_TYPE
    attention
  GENERATOR
    "generate_e2e_attention_tests.py"
  GENERATOR_ARGS
    "--query_type=f16"
    "--key_type=f16"
    "--value_type=f16"
    "--shapes=large"
  TEST_RUNNER
    iree_tools_testing_e2e_iree-e2e-attention-test
  TARGET_BACKENDS
    "llvm-cpu"
  DRIVERS
    "local-task"
  LABELS
    "hostonly"
    "local"
)

# To distinguish between CDNA(gfx9) and RDNA3(gfx11)
if(IREE_HIP_TEST_TARGET_CHIP MATCHES "^gfx9")

unset(IREE_HIP_TEST_COMPILER_FLAGS)
list(APPEND IREE_HIP_TEST_COMPILER_FLAGS
  "--iree-hip-target=${IREE_HIP_TEST_TARGET_CHIP}"
)

iree_generated_e2e_runner_test(
  NAME
    e2e_attention_gpu_cdna3_f16_f16_f16_small
  TEST_TYPE
    attention
  GENERATOR
    "generate_e2e_attention_tests.py"
  GENERATOR_ARGS
    "--query_type=f16"
    "--key_type=f16"
    "--value_type=f16"
    "--shapes=small"
  TEST_RUNNER
    iree_tools_testing_e2e_iree-e2e-attention-test
  TARGET_BACKENDS
    "rocm"
  DRIVERS
    "hip"
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  LABELS
    "noasan"
    "nomsan"
    "notsan"
    "noubsan"
    "requires-gpu-cdna3"
)

iree_generated_e2e_runner_test(
  NAME
    e2e_attention_gpu_cdna3_f16_f16_f16_medium
  TEST_TYPE
    attention
  GENERATOR
    "generate_e2e_attention_tests.py"
  GENERATOR_ARGS
    "--query_type=f16"
    "--key_type=f16"
    "--value_type=f16"
    "--shapes=medium"
  TEST_RUNNER
    iree_tools_testing_e2e_iree-e2e-attention-test
  TARGET_BACKENDS
    "rocm"
  DRIVERS
    "hip"
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  LABELS
    "noasan"
    "nomsan"
    "notsan"
    "noubsan"
    "requires-gpu-cdna3"
)

iree_generated_e2e_runner_test(
  NAME
    e2e_attention_gpu_cdna3_f16_f16_f16_large
  TEST_TYPE
    attention
  GENERATOR
    "generate_e2e_attention_tests.py"
  GENERATOR_ARGS
    "--query_type=f16"
    "--key_type=f16"
    "--value_type=f16"
    "--shapes=large"
  TEST_RUNNER
    iree_tools_testing_e2e_iree-e2e-attention-test
  TARGET_BACKENDS
    "rocm"
  DRIVERS
    "hip"
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  LABELS
    "noasan"
    "nomsan"
    "notsan"
    "noubsan"
    "requires-gpu-cdna3"
)
endif()
