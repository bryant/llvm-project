add_llvm_library(LLVMAVRAsmParser
  AVRAsmParser.cpp

  LINK_COMPONENTS

  AVRDesc
  AVRInfo
  MC
  MCParser
  Support
)
