import unittest

from src.wifi_scanner import (
    _merge_windows_interface_details,
    _parse_windows,
    _parse_windows_interfaces,
)


class WindowsWifiScannerTest(unittest.TestCase):
    def test_merges_connected_interface_details_into_visible_network(self):
        scan_output = """\
SSID 1 : Airtel_NITHISH0002
    Network type            : Infrastructure
    Authentication          : WPA2-Personal
    Encryption               : CCMP
"""
        interface_output = """\
Name                   : Wi-Fi
Description            : Wireless adapter
State                  : connected
SSID                   : Airtel_NITHISH0002
AP BSSID               : 30:15:77:d9:b5:87
Channel                : 36
Signal                 : 87%
Rssi                   : -41
"""

        networks = _parse_windows(scan_output)
        interfaces = _parse_windows_interfaces(interface_output)
        merged = _merge_windows_interface_details(networks, interfaces)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["connection_status"], "Connected")
        self.assertEqual(merged[0]["bssid"], "30:15:77:d9:b5:87")
        self.assertEqual(merged[0]["signal"], -41)
        self.assertEqual(merged[0]["signal_percent"], 87)
        self.assertEqual(merged[0]["channel"], 36)

    def test_does_not_add_disconnected_interfaces_as_nearby_networks(self):
        interface_output = """\
Name                   : Wi-Fi
State                  : disconnected
SSID                   : OldNetwork
"""

        self.assertEqual(_parse_windows_interfaces(interface_output), [])
        self.assertEqual(
            _merge_windows_interface_details([], _parse_windows_interfaces(interface_output)),
            [],
        )

    def test_keeps_visible_networks_when_no_interface_is_connected(self):
        scan_output = """\
SSID 1 : Guest
    Authentication          : WPA2-Personal
"""

        networks = _parse_windows(scan_output)
        merged = _merge_windows_interface_details(networks, [])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["ssid"], "Guest")
        self.assertNotIn("connection_status", merged[0])


if __name__ == "__main__":
    unittest.main()
