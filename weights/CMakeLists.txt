cmake_minimum_required(VERSION 3.14)

project(weights_download)

function(download_file url filename destination)
    # Ensure the destination directory exists
    file(MAKE_DIRECTORY ${destination})
    set(file_path "${destination}/${filename}")

    if(NOT EXISTS "${file_path}")
        # Download the file
        file(DOWNLOAD ${url} ${file_path}
            STATUS download_status
            SHOW_PROGRESS)

        # Check the download status
        list(GET download_status 0 status_code)
        if(status_code EQUAL 0)
            message(STATUS "File downloaded successfully: ${file_path}")
        else()
            message(FATAL_ERROR "Error downloading file: ${url}")
        endif()

    endif()

    # Install the downloaded file (optional)
    install(FILES ${file_path} DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/weights/" )
endfunction()

# Example usage of the function to download multiple files
set(DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloaded_files")

download_file("http://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/3DAqcSkNNrkXeQ" "mask_rcnn_green.h5" ${DOWNLOAD_DIR})
download_file("http://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/8_V0Z31zmLoiNA" "mask_rcnn_hydro.h5" ${DOWNLOAD_DIR})
download_file("http://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/NQcQ4NlAZT5ykw" "mask_rcnn_coco.h5" ${DOWNLOAD_DIR})
