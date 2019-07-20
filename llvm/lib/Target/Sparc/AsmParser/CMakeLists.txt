set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  SparcDesc
  SparcInfo
  Support
  )

add_llvm_library(LLVMSparcAsmParser
  SparcAsmParser.cpp
  )
