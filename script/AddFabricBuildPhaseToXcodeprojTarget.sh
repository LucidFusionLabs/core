#!/bin/sh
echo Adding Fabric Run Script Build Phase to $2 target $3 for API token $CRASHLYTICS_API_TOKEN on $1
core/script/AddRunScriptBuildPhaseToXcodeprojTarget.rb $2 $3 "\$PROJECT_DIR/core/imports/fabric-$1/Crashlytics.framework/run $CRASHLYTICS_API_TOKEN $CRASHLYTICS_BUILD_SECRET"
