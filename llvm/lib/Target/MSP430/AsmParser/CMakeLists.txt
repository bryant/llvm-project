set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  MSP430Desc
  MSP430Info
  Support
  )

add_llvm_library(LLVMMSP430AsmParser
  MSP430AsmParser.cpp
)
