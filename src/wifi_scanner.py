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
        if re.match(r"^SSID\s+\d+\s*:", line, re.IGNORECASE):
            if current and current.get("ssid"):
                networks.append(current)
            current = {"ssid": re.split(r"\s*:\s*", line, maxsplit=1)[1], "bssid": "", "signal": None, "signal_percent": None, "channel": None, "security": "Unknown"}
        elif current is not None and re.match(r"^BSSID\s+\d+\s*:", line, re.IGNORECASE):
            current["bssid"] = re.split(r"\s*:\s*", line, maxsplit=1)[1]
        elif current is not None and re.match(r"^Signal\s*:", line, re.IGNORECASE):
            value = re.split(r"\s*:\s*", line, maxsplit=1)[1].rstrip("%").strip()
            match = re.search(r"\d+", value)
            current["signal_percent"] = int(match.group()) if match else None
            current["signal"] = round(-100 + current["signal_percent"] * 0.8, 1) if match else None
        elif current is not None and re.match(r"^Channel\s*:", line, re.IGNORECASE):
            value = re.split(r"\s*:\s*", line, maxsplit=1)[1]
            match = re.search(r"\d+", value)
            current["channel"] = int(match.group()) if match else None
        elif current is not None and re.match(r"^Authentication\s*:", line, re.IGNORECASE):
            current["security"] = re.split(r"\s*:\s*", line, maxsplit=1)[1].strip()
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
        networks.append({"ssid": ssid or "<hidden>", "bssid": bssid, "signal": round(-100 + signal_value * 0.8, 1) if signal_value is not None else None, "signal_percent": signal_value,
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