add_llvm_library(LLVMLTO
  Caching.cpp
  LTO.cpp
  LTOBackend.cpp
  LTOModule.cpp
  LTOCodeGenerator.cpp
  SummaryBasedOptimizations.cpp
  UpdateCompilerUsed.cpp
  ThinLTOCodeGenerator.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/LTO

  DEPENDS
  intrinsics_gen
  llvm_vcsrevision_h

  LINK_COMPONENTS

  AggressiveInstCombine
  Analysis
  BitReader
  BitWriter
  CodeGen
  Core
  InstCombine
  Linker
  MC
  ObjCARCOpts
  Object
  Passes
  Remarks
  ScalarOpts
  Support
  Target
  TransformUtils
  ipo
)
