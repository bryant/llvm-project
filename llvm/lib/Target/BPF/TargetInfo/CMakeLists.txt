set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(LLVMBPFInfo
  BPFTargetInfo.cpp
  )
