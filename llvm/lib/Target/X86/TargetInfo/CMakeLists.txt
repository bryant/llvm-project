set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(LLVMX86Info
  X86TargetInfo.cpp
  )
