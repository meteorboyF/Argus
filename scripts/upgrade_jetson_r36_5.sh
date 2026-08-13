#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_FILE=/etc/apt/sources.list.d/nvidia-l4t-apt-source.list
BACKUP_DIR=/var/backups/argus-r36.4.7-to-r36.5

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this script with sudo:" >&2
  echo "  sudo bash $0" >&2
  exit 1
fi

if ! grep -q '^# R36 (release), REVISION: 4\.7' /etc/nv_tegra_release; then
  echo "Refusing: this device is not reporting Jetson Linux R36.4.7." >&2
  cat /etc/nv_tegra_release >&2
  exit 1
fi

if [[ ! -f ${SOURCE_FILE} ]]; then
  echo "Refusing: ${SOURCE_FILE} does not exist." >&2
  exit 1
fi

mkdir -p "${BACKUP_DIR}"
cp -a /etc/nv_tegra_release "${BACKUP_DIR}/nv_tegra_release.before"
cp -a "${SOURCE_FILE}" "${BACKUP_DIR}/nvidia-l4t-apt-source.list.before"
dpkg-query -W > "${BACKUP_DIR}/dpkg-packages.before.txt"
uname -a > "${BACKUP_DIR}/uname.before.txt"

# Change only NVIDIA Jetson repository entries. Ubuntu and third-party sources
# are deliberately untouched.
sed -i -E '/repo\.download\.nvidia\.com\/jetson\// s/[[:space:]]r36\.4[[:space:]]/ r36.5 /' "${SOURCE_FILE}"

if grep -qE 'repo\.download\.nvidia\.com/jetson/.*[[:space:]]r36\.4[[:space:]]' "${SOURCE_FILE}"; then
  echo "Refusing: an NVIDIA Jetson repository still points at r36.4." >&2
  cp -a "${BACKUP_DIR}/nvidia-l4t-apt-source.list.before" "${SOURCE_FILE}"
  exit 1
fi

echo "NVIDIA repository configuration now reads:"
grep 'repo.download.nvidia.com/jetson/' "${SOURCE_FILE}"

apt-get update

candidate=$(apt-cache policy nvidia-l4t-core | awk '/Candidate:/ {print $2}')
echo "nvidia-l4t-core candidate: ${candidate}"
case "${candidate}" in
  36.5*) ;;
  *)
    echo "Refusing upgrade: repository did not offer an R36.5 candidate." >&2
    cp -a "${BACKUP_DIR}/nvidia-l4t-apt-source.list.before" "${SOURCE_FILE}"
    apt-get update
    exit 1
    ;;
esac

apt-get dist-upgrade

echo
echo "Package installation finished. DO NOT start ARGUS yet."
echo "When ready, reboot manually with: sudo reboot"
