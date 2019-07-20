set(LLVM_LINK_COMPONENTS
  HexagonDesc
  HexagonInfo
  MC
  MCDisassembler
  Support
  )

add_llvm_library(LLVMHexagonDisassembler
  HexagonDisassembler.cpp
  )
