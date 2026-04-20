if(NOT DEFINED INPUT_FILE OR NOT DEFINED OUTPUT_FILE)
  message(FATAL_ERROR "INPUT_FILE and OUTPUT_FILE are required")
endif()

file(STRINGS "${INPUT_FILE}" _lines NEWLINE_CONSUME)

set(_collect FALSE)
set(_output "")

foreach(_line IN LISTS _lines)
  if(NOT _collect)
    if(_line STREQUAL "const char* kLensDiffMetalSource = R\"METAL(")
      set(_collect TRUE)
    endif()
  else()
    if(_line STREQUAL ")METAL\";")
      set(_collect FALSE)
      break()
    endif()
    string(APPEND _output "${_line}\n")
  endif()
endforeach()

if(_collect)
  message(FATAL_ERROR "Failed to find end of kLensDiffMetalSource block in ${INPUT_FILE}")
endif()

if(_output STREQUAL "")
  message(FATAL_ERROR "Failed to extract Metal source from ${INPUT_FILE}")
endif()

file(WRITE "${OUTPUT_FILE}" "${_output}")
