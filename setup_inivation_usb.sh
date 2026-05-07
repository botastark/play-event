#!/usr/bin/env bash
set -e

RULE_FILE="/etc/udev/rules.d/99-inivation-davis.rules"
VENDOR_ID="152a"
PRODUCT_ID="841a"
GROUP_NAME="plugdev"

echo "Creating udev rule for iniVation DAViS (VID:PID=${VENDOR_ID}:${PRODUCT_ID})..."

# 1) Ensure group exists (plugdev usually exists on Ubuntu, but this is safe)
if ! getent group "${GROUP_NAME}" >/dev/null; then
  echo "Creating group ${GROUP_NAME}..."
  sudo groupadd "${GROUP_NAME}"
fi

# 2) Add current user to group
echo "Adding user ${USER} to group ${GROUP_NAME}..."
sudo usermod -aG "${GROUP_NAME}" "${USER}"

# 3) Write udev rule
echo "Writing udev rule to ${RULE_FILE}..."
sudo bash -c "cat > ${RULE_FILE}" <<EOF
SUBSYSTEM=="usb", ATTR{idVendor}=="${VENDOR_ID}", ATTR{idProduct}=="${PRODUCT_ID}", GROUP="${GROUP_NAME}", MODE="0660"
EOF

# 4) Reload udev rules and trigger
echo "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo
echo "Done. Now:"
echo "  1) Unplug and replug the iniVation camera."
echo "  2) Log out and back in (or reboot) so your new group membership takes effect."
echo "  3) Verify with: ls -l /dev/bus/usb/*/* | grep ${VENDOR_ID}"
echo
echo "After that, you should be able to run dv-processing scripts without sudo."