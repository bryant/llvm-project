include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMLanaiAsmParser
  LanaiAsmParser.cpp

  LINK_COMPONENTS

  LanaiDesc
  LanaiInfo
  MC
  MCParser
  Support
  )

add_dependencies( LLVMLanaiAsmParser LanaiCommonTableGen )
