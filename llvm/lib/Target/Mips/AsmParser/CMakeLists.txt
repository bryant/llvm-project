set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  MipsDesc
  MipsInfo
  Support
  )

add_llvm_library(LLVMMipsAsmParser
  MipsAsmParser.cpp
  )
