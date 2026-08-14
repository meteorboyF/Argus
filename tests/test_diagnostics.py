from argus.diagnostics import has_usb_audio, parse_gr3d_samples, parse_power_mode, parse_v4l2_names


def test_parse_power_mode():
    assert parse_power_mode("NV Power Mode: MAXN_SUPER\n2") == "MAXN_SUPER"
    assert parse_power_mode("unavailable") is None


def test_parse_tegrastats_gr3d_samples():
    text = "RAM 1/2 CPU [1] GR3D_FREQ 0%\nRAM 1/2 GR3D_FREQ 87%@[612]"
    assert parse_gr3d_samples(text) == [0, 87]


def test_parse_v4l2_device_names_ignores_nodes():
    text = "Arducam B0495:\n\t/dev/video0\nArducam B0459:\n\t/dev/video4\n"
    assert parse_v4l2_names(text) == ["Arducam B0495", "Arducam B0459"]


def test_usb_audio_requires_actual_usb_device():
    assert has_usb_audio("card 2: Device [USB Audio Device], device 0: USB Audio")
    assert not has_usb_audio("card 0: HDA [NVIDIA Jetson Orin Nano HDA]")
