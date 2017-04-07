#!/bin/sh
echo Adding Fabric Run Script Build Phase to $1 target $2 for API token $CRASHLYTICS_API_TOKEN
core/script/AddRunScriptBuildPhaseToXcodeprojTarget.rb $1 $2 "\$PROJECT_DIR/core/imports/fabric-ios/Crashlytics.framework/run $CRASHLYTICS_API_TOKEN $CRASHLYTICS_BUILD_SECRET"
