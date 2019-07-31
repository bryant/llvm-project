add_llvm_library(LLVMMSP430Disassembler
  MSP430Disassembler.cpp

  LINK_COMPONENTS

  MCDisassembler
  MSP430Info
  Support
  )
