set(LLVM_LINK_COMPONENTS
  MCDisassembler
  RISCVInfo
  Support
  )

add_llvm_library(LLVMRISCVDisassembler
  RISCVDisassembler.cpp
  )
