set(LLVM_LINK_COMPONENTS
  AVRDesc
  AVRInfo
  MC
  MCParser
  Support
  )

add_llvm_library(LLVMAVRAsmParser
  AVRAsmParser.cpp
)
