"""Extract spectra from Living Optics ``.lo`` files with the Living Optics SDK (not on PyPI; proprietary).

The SDK is imported lazily, so every other part of the package works without it. Requires an SDK distribution that provides
``lo.sdk.api.acquisition`` with ``SpectralDecoder``, and the factory calibration folder for the camera.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger("hsiplot")

_HELP = ("The Living Optics SDK was not found or does not provide the spectral decoder. Install an SDK distribution that includes "
         "`lo.sdk.api.acquisition.data.decode.SpectralDecoder` (account at https://cloud.livingoptics.com) into the same environment.")


def _sdk():
    try:
        from lo.sdk.api.acquisition.data.decode import SpectralDecoder
        from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
        from lo.sdk.api.acquisition.io.open import open as sdkopen
    except ImportError as e:  # pragma: no cover - depends on the user's environment
        raise ImportError(f"{e}. {_HELP}") from e
    return sdkopen, SpectralDecoder, LORAWtoRGB8


def _decoder(spectral_decoder, calibration_folder: Optional[str], field_calibration: Optional[str]):
    """Build the decoder the way the official Living Optics examples do:
    ``SpectralDecoder.from_calibration(factory_calibration_folder, field_calibration_file)`` (field calibration may be ``None``)."""
    if not calibration_folder:
        raise ValueError("calibration_folder is required: the factory calibration folder supplied by Living Optics for your camera")
    return spectral_decoder.from_calibration(str(calibration_folder), str(field_calibration) if field_calibration else None)


def decode_frame(lo_file, calibration_folder: str, frame: int = 0, field_calibration: Optional[str] = None) -> Dict:
    """Decode one frame. Returns ``spectra`` (N, bands), ``coords`` (N, 2), ``wavelengths``, ``scene`` (RGB)."""
    sdkopen, SpectralDecoder, LORAWtoRGB8 = _sdk()
    with sdkopen(str(lo_file)) as f:
        n = len(f)
        if not 0 <= frame < n:
            raise ValueError(f"{lo_file} has {n} frame(s); frame {frame} is out of range")
        dec = _decoder(SpectralDecoder, calibration_folder, field_calibration)
        f.seek(frame)
        raw = f.read()
        metadata, scene, spectra = dec(raw, LORAWtoRGB8)
    return {"spectra": np.asarray(spectra, dtype=np.float32), "coords": np.asarray(metadata.sampling_coordinates),
            "wavelengths": np.asarray(metadata.wavelengths, dtype=float), "scene": np.asarray(scene), "n_frames": n}


def extract_to_npz(lo_file, out, calibration_folder: str, frame: int = 0, field_calibration: Optional[str] = None,
                   white_reference: Optional[str] = None, dark_reference: Optional[str] = None) -> Path:
    """Decode a frame and save ``spectra, coords, wavelengths, quantity`` to ``out`` (.npz).

    With ``white_reference`` (a ``.npy`` made by ``hsiplot reference``) the spectra are converted to reflectance and
    ``quantity`` is ``reflectance``; otherwise they are radiance.
    """
    from .calibrate import to_reflectance
    d = decode_frame(lo_file, calibration_folder, frame, field_calibration)
    S, quantity = d["spectra"], "radiance"
    if white_reference:
        dark = np.load(dark_reference) if dark_reference else None
        S, quantity = to_reflectance(S, np.load(white_reference), dark).astype(np.float32), "reflectance"
    elif dark_reference:
        raise ValueError("dark_reference is only used together with white_reference")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, spectra=S, coords=d["coords"], wavelengths=d["wavelengths"], quantity=quantity)
    log.info("%s frame %d: %d spectra x %d bands (%s) -> %s", lo_file, frame, S.shape[0], S.shape[1], quantity, out)
    return out


def check_file(lo_file, calibration_folder: Optional[str] = None, frame: int = 0, field_calibration: Optional[str] = None) -> Dict:
    """Step-by-step SDK check that reports which step failed. Returns a dict of results; raises nothing."""
    rep: Dict = {"steps": []}

    def step(name, fn):
        try:
            r = fn()
            rep["steps"].append((name, "ok"))
            return r
        except Exception as e:  # noqa: BLE001
            rep["steps"].append((name, f"FAILED: {e!r}"))
            return None

    mods = step("import SDK (open, SpectralDecoder, LORAWtoRGB8)", _sdk)
    if mods is None:
        rep["hint"] = _HELP
        return rep
    d = step("open, build decoder, read and decode frame", lambda: decode_frame(lo_file, calibration_folder, frame, field_calibration))
    if d is not None:
        S = d["spectra"]
        rep.update(n_spectra=int(S.shape[0]), n_bands=int(S.shape[1]), n_frames=int(d["n_frames"]),
                   wavelength_range=[float(d["wavelengths"][0]), float(d["wavelengths"][-1])],
                   value_range=[float(S.min()), float(S.max())], any_nan=bool(np.isnan(S).any()),
                   fraction_nonpositive=float((S <= 0).any(1).mean()))
    return rep
