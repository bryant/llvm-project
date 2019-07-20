set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(LLVMHexagonInfo
  HexagonTargetInfo.cpp
  )
