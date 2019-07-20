set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(LLVMMipsInfo
  MipsTargetInfo.cpp
  )
