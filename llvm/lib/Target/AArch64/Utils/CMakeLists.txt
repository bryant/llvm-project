set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(LLVMAArch64Utils
  AArch64BaseInfo.cpp
  )
