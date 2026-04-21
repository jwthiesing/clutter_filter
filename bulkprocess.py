#!/usr/bin/env python3
"""Bulk process RKC IQ files to NetCDF moment files."""

import argparse
import glob
import os
import sys

import numpy as np

from clutter_filter import GroundClutterFilter
import radarkitIQ as radarkit
import momentgen as mg
from raxpolCf import raxpolCf


def process_file(fname, count, grf, outdir):
    print(f"Processing: {fname}")

    rkid = radarkit.rkcfile(fname)
    pulses_raw = rkid.pulses['iq']

    # (pulse, iq, gate, hv) -> (gate, pulse, hv)
    pulses_raw = pulses_raw.transpose((1, 2, 3, 0))
    pulses = pulses_raw[0, :, :, :] + 1j * pulses_raw[1, :, :, :]
    pulses = pulses.transpose((0, 2, 1)).astype(np.complex128)
    # (gate, pulse, hv)

    lamb = rkid.header['desc']['wavelength']
    prt = rkid.header['config']['prt']
    sample_freq = int(rkid.header['waveform']['fs'])
    sample_time = 1 / sample_freq
    va = lamb / (4 * prt)

    totaln = pulses.shape[1]
    rayCount = int(np.floor(totaln / count))
    gateCount = pulses.shape[0]
    n = rayCount * count

    az = rkid.pulses['azimuthDegrees']
    el = rkid.pulses['elevationDegrees']

    X_h = pulses[:, :n, 0].reshape(gateCount, rayCount, count)
    X_v = pulses[:, :n, 1].reshape(gateCount, rayCount, count)

    az = az[:n].reshape(rayCount, count)[:, 0]
    el_2d = el[:n].reshape(rayCount, count)
    el = np.nanmean(el_2d, axis=1)

    if grf:
        filter_inst = GroundClutterFilter(
            wavelength=lamb,
            scan_rate=1 / (count * prt),
            prt=prt,
            num_samples=count,
        )

        print("  Filtering H polarization...")
        fi_h, fq_h, *_ = filter_inst.filter_iq_data(
            X_h.real, X_h.imag, cnr_db_map=None, apply_interpolation=True
        )
        X_h = fi_h + 1j * fq_h
        del fi_h, fq_h

        print("  Filtering V polarization...")
        fi_v, fq_v, *_ = filter_inst.filter_iq_data(
            X_v.real, X_v.imag, cnr_db_map=None, apply_interpolation=True
        )
        X_v = fi_v + 1j * fq_v
        del fi_v, fq_v

    noise = rkid.header['config']['noise']
    N_h, N_v = noise

    if rkid.header['buildNo'] >= 4:
        if rkid.header['dataType'] == 'raw':
            dr = rkid.header['config']['pulseGateSize']
        elif rkid.header['dataType'] == 'compressed':
            dr = rkid.header['desc']['pulseToRayRatio'] * rkid.header['config']['pulseGateSize']
        else:
            print("  Warning: unknown dataType, defaulting dr=30")
            dr = 30.0
    else:
        dr = 30.0

    R = (np.arange(0, gateCount) + 0.5) * dr
    C = rkid.header['config']['ZCal']
    Cd = rkid.header['config']['DCal']
    Cp = rkid.header['config']['PCal']

    moments = mg.get_moments(X_h, X_v, N_h, N_v, R, va, C, Cd, Cp)

    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(fname))[0]
    suffix = '-f.nc' if grf else '.nc'
    fname_out = os.path.join(outdir, base + suffix)

    rcf = raxpolCf()
    rcf.setVolume()
    rcf.setSweep()
    rcf.setTime((rkid.pulses['time_tv_sec'][:n:count]).astype(np.float64))
    rcf.setRange(R.astype(np.float32))
    rcf.setPosition(
        np.nanmean(rkid.header['desc']['latitude']),
        np.nanmean(rkid.header['desc']['longitude']),
    )
    rcf.setScanningStrategy('ppi')
    rcf.setTargetAngle(np.nanmean(rkid.pulses['elevationDegrees']))
    rcf.setAzimuth(az)
    rcf.setElevation(el)
    rcf.setPulseWidthSeconds(
        (rkid.pulses['pulseWidthSampleCount'][:n:count] * sample_time).astype(np.float32)
    )
    rcf.setPrtSeconds(np.tile(prt, (rayCount, 1)))
    rcf.setWavelengthMeters(np.tile(lamb, (rayCount, 1)))

    rcf.setDBZ(moments['DBZ'])
    rcf.setVEL(moments['VEL'])
    rcf.setWIDTH(moments['WIDTH'])
    rcf.setZDR(moments['ZDR'])
    rcf.setRHOHV(moments['RHOHV'])
    rcf.setPHIDP(moments['PHIDP'], units='degrees')
    rcf.setSNRH(moments['SNRH'])
    rcf.setSNRV(moments['SNRV'])

    rcf.saveToFile(fname_out)
    print(f"  Saved: {fname_out}")
    return fname_out


def main():
    parser = argparse.ArgumentParser(
        description='Bulk process RKC IQ files to NetCDF moment files.',
    )
    parser.add_argument(
        'pattern',
        nargs='+',
        help='File(s) or glob pattern(s) to process (e.g. data/*.rkc); use -r to search subfolders',
    )
    parser.add_argument(
        '-n', '--count',
        type=int,
        default=50,
        metavar='PULSES',
        help='Pulses per ray (default: 50)',
    )
    parser.add_argument(
        '--grf',
        action='store_true',
        default=False,
        help='Apply ground clutter filter (GRF)',
    )
    parser.add_argument(
        '-o', '--outdir',
        default='netcdf',
        metavar='DIR',
        help='Output subdirectory for NetCDF files (default: netcdf)',
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        default=False,
        help='Search subdirectories recursively for each pattern',
    )
    args = parser.parse_args()

    files = []
    for pat in args.pattern:
        if args.recursive and '**' not in pat:
            dirpart, filepart = os.path.split(pat)
            pat = os.path.join(dirpart, '**', filepart) if dirpart else os.path.join('**', filepart)
        matched = sorted(glob.glob(pat, recursive=True))
        if not matched:
            print(f"Warning: no files matched pattern '{pat}'", file=sys.stderr)
        files.extend(matched)

    if not files:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} file(s). count={args.count}, grf={args.grf}, outdir='{args.outdir}'")

    failed = []
    for fname in files:
        try:
            process_file(fname, args.count, args.grf, args.outdir)
        except Exception as exc:
            print(f"  ERROR processing {fname}: {exc}", file=sys.stderr)
            failed.append(fname)

    print(f"\nDone. {len(files) - len(failed)}/{len(files)} succeeded.")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)


if __name__ == '__main__':
    main()
