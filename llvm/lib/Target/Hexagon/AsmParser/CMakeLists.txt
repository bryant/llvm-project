include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMHexagonAsmParser
  HexagonAsmParser.cpp

  LINK_COMPONENTS

  HexagonDesc
  HexagonInfo
  MC
  MCParser
  Support
  )

add_dependencies( LLVMHexagonAsmParser HexagonCommonTableGen )
