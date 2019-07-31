add_llvm_library(LLVMMSP430AsmParser
  MSP430AsmParser.cpp

  LINK_COMPONENTS

  MC
  MCParser
  MSP430Desc
  MSP430Info
  Support
)
