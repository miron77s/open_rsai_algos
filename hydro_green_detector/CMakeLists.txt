cmake_minimum_required(VERSION 3.5)

install(DIRECTORY ./
       DESTINATION bin/
       FILES_MATCHING
       PATTERN "*.py"
       PATTERN "*.sh")
