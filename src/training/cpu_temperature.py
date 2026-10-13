"""Read existing Windows hardware-monitor WMI providers; never install drivers."""

from __future__ import annotations

import json
import math
import platform
import re
import subprocess
import threading
from datetime import UTC, datetime

PROVIDERS = ("root\\LibreHardwareMonitor", "root\\OpenHardwareMonitor")


def read_cpu_temperatures(provider):
    if provider not in PROVIDERS:
        raise ValueError("Unknown CPU telemetry provider")
    # Sensor.Parent must join an explicitly typed CPU Hardware.Identifier.
    # Board/ACPI/GPU zones are never relabeled as CPU measurements.
    command = (
        "$ErrorActionPreference='Stop'; "
        f"$hardware=@(Get-CimInstance -Namespace '{provider}' -ClassName Hardware "
        "-OperationTimeoutSec 3 | Where-Object HardwareType -eq 'Cpu'); "
        f"$sensors=@(Get-CimInstance -Namespace '{provider}' -ClassName Sensor "
        "-OperationTimeoutSec 3 | Where-Object SensorType -eq 'Temperature'); "
        "@($sensors | Where-Object { $hardware.Identifier -contains $_.Parent } "
        "| Select-Object SensorType,Value,Parent) | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=8,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    rows = json.loads(result.stdout) if result.stdout.strip() else []
    rows = rows if isinstance(rows, list) else [rows]
    temperatures = []
    for row in rows:
        value = row.get("Value")
        if (
            row.get("SensorType") == "Temperature"
            and re.fullmatch(r"/(?:intelcpu|amdcpu)/[0-9]+", str(row.get("Parent", "")))
            and type(value) in {int, float}
            and math.isfinite(value)
            and 0 < value < 150
        ):
            temperatures.append(float(value))
    return temperatures


class CPUTemperatureMonitor:
    """Best-effort CPU sensor polling. Unavailable telemetry is explicit, not zero."""

    def __init__(self, *, enabled=True):
        self.enabled = enabled
        self.provider = None
        self.samples = []
        self.unavailable_reason = "not_sampled"
        self.stop_event = threading.Event()
        self.thread = None

    def sample(self):
        if not self.enabled:
            self.unavailable_reason = "disabled_for_fixture"
            return
        if platform.system() != "Windows":
            self.unavailable_reason = "windows_provider_not_applicable"
            return
        for provider in [self.provider] if self.provider else PROVIDERS:
            try:
                values = read_cpu_temperatures(provider)
            except (OSError, subprocess.SubprocessError, ValueError, TypeError):
                continue
            if values:
                self.provider = provider
                self.samples.append(
                    {
                        "time_utc": datetime.now(UTC).isoformat(),
                        "temperature_c": max(values),
                        "sensor_count": len(values),
                    }
                )
                self.unavailable_reason = None
                return
        self.unavailable_reason = "no_readable_cpu_temperature_sensor"

    def start(self):
        self.sample()
        # Do not repeatedly spawn commands against missing providers.
        if self.unavailable_reason:
            return

        def poll():
            while not self.stop_event.wait(5):
                self.sample()

        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=9)

    def receipt(self):
        return {
            "provider": self.provider,
            "samples": list(self.samples),
            "unavailable_reason": self.unavailable_reason,
            "temperature_available": bool(self.samples)
            and self.unavailable_reason is None,
            "scope": "maximum_of_provider_identified_cpu_temperature_sensors",
            "independent_sensor_freshness_verified": False,
            "cpu_temperature_stop_threshold": None,
            "limitation": "GPU telemetry does not establish CPU or whole-laptop thermal safety",
        }
