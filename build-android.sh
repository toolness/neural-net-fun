#! /bin/bash
docker-compose run --rm app cargo quad-apk build --release

# Note that this will fail if the app isn't already installed,
# but we need to run it or else `adb install` might not work
# when the app *is* already installed.
adb uninstall com.toolness.neural_net_fun

adb install target/android-artifacts/release/apk/neural-net-fun.apk
