import platform
import re
import shutil
import subprocess


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Wi-Fi scan command failed")
    return result.stdout


def _parse_windows(output):
    networks = []
    current = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("SSID ") and ":" in line:
            if current and current.get("ssid"):
                networks.append(current)
            current = {"ssid": line.split(":", 1)[1].strip(), "bssid": "", "signal": None, "channel": None, "security": "Unknown"}
        elif current is not None and line.startswith("BSSID ") and ":" in line:
            current["bssid"] = line.split(":", 1)[1].strip()
        elif current is not None and line.startswith("Signal") and ":" in line:
            value = line.split(":", 1)[1].strip().rstrip("%").strip()
            current["signal"] = int(value) if value.isdigit() else None
        elif current is not None and line.startswith("Channel") and ":" in line:
            value = line.split(":", 1)[1].strip()
            current["channel"] = int(value) if value.isdigit() else None
        elif current is not None and line.startswith("Authentication") and ":" in line:
            current["security"] = line.split(":", 1)[1].strip()
    if current and current.get("ssid"):
        networks.append(current)
    return networks


def _parse_linux(output):
    networks = []
    for line in output.splitlines():
        parts = line.split(":")
        if len(parts) < 5:
            continue
        ssid, bssid, signal, channel, security = [part.strip() for part in parts[:5]]
        try:
            signal_value = int(signal)
        except ValueError:
            signal_value = None
        networks.append({"ssid": ssid or "<hidden>", "bssid": bssid, "signal": signal_value,
                         "channel": int(channel) if channel.isdigit() else None, "security": security or "Unknown"})
    return networks


def scan_wifi_networks():
    """Return only fields exposed by the local OS Wi-Fi tooling."""
    if platform.system() == "Windows":
        if not shutil.which("netsh"):
            raise RuntimeError("Windows netsh is unavailable")
        return _parse_windows(_run(["netsh", "wlan", "show", "networks", "mode=bssid"]))
    if platform.system() == "Linux":
        if not shutil.which("nmcli"):
            raise RuntimeError("nmcli is unavailable; install NetworkManager")
        output = _run(["nmcli", "-t", "-f", "SSID,BSSID,SIGNAL,CHAN,SECURITY", "device", "wifi", "list"])
        return _parse_linux(output)
    raise RuntimeError(f"Unsupported operating system: {platform.system()}")