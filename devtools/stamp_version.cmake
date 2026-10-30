# Write the build stamp of the plugin (openmmapi/src/InteraxPluginVersion.cpp.in -> OUTPUT): VERSION and the
# "git describe --always --dirty" of SOURCE_DIR. Run at configure time and before every build of the library;
# configure_file() rewrites OUTPUT only when its content changes, so an unchanged commit does not trigger a rebuild.
EXECUTE_PROCESS(COMMAND git describe --always --dirty --abbrev=12
                WORKING_DIRECTORY ${SOURCE_DIR}
                OUTPUT_VARIABLE INTERAX_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
IF(NOT INTERAX_GIT_COMMIT)
    SET(INTERAX_GIT_COMMIT "unknown")
ENDIF()
SET(INTERAX_PLUGIN_VERSION ${VERSION})
CONFIGURE_FILE(${SOURCE_DIR}/openmmapi/src/InteraxPluginVersion.cpp.in ${OUTPUT})
