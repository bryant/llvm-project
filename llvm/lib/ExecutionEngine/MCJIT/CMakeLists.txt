set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  Object
  RuntimeDyld
  Support
  Target
  )

add_llvm_library(LLVMMCJIT
  MCJIT.cpp

  DEPENDS
  intrinsics_gen
  )
