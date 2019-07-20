set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(LLVMRISCVInfo
  RISCVTargetInfo.cpp
  )
