set(LLVM_LINK_COMPONENTS
  CodeGen
  Core
  DebugInfoDWARF
  ExecutionEngine
  Object
  Support
  )

add_llvm_library(LLVMPerfJITEvents
  PerfJITEventListener.cpp
  )

add_dependencies(LLVMPerfJITEvents LLVMCodeGen)
