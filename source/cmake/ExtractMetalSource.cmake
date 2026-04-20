if(NOT DEFINED INPUT_FILE OR NOT DEFINED OUTPUT_FILE)
  message(FATAL_ERROR "INPUT_FILE and OUTPUT_FILE are required")
endif()

file(READ "${INPUT_FILE}" _content)

set(_start_marker "R\"METAL(")
set(_end_marker ")METAL\";")

string(FIND "${_content}" "${_start_marker}" _start_index)
if(_start_index EQUAL -1)
  message(FATAL_ERROR "Failed to find Metal source start marker in ${INPUT_FILE}")
endif()

string(LENGTH "${_start_marker}" _start_marker_length)
math(EXPR _payload_start "${_start_index} + ${_start_marker_length}")

string(SUBSTRING "${_content}" "${_payload_start}" 1 _first_char)
if(_first_char STREQUAL "\r")
  math(EXPR _payload_start "${_payload_start} + 1")
  string(SUBSTRING "${_content}" "${_payload_start}" 1 _first_char)
endif()
if(_first_char STREQUAL "\n")
  math(EXPR _payload_start "${_payload_start} + 1")
endif()

string(FIND "${_content}" "${_end_marker}" _end_index REVERSE)
if(_end_index EQUAL -1 OR _end_index LESS _payload_start)
  message(FATAL_ERROR "Failed to find Metal source end marker in ${INPUT_FILE}")
endif()

math(EXPR _payload_length "${_end_index} - ${_payload_start}")
string(SUBSTRING "${_content}" "${_payload_start}" "${_payload_length}" _output)

if(_output STREQUAL "")
  message(FATAL_ERROR "Failed to extract Metal source from ${INPUT_FILE}")
endif()

file(WRITE "${OUTPUT_FILE}" "${_output}")
