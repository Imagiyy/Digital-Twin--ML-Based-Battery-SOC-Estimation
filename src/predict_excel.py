"""CLI Tool for Testing Battery SOC Estimation Using Excel (.xlsx / .xls) and CSV Files.

Usage examples:
    # 1. Run inference on an Excel file:
    python src/predict_excel.py sample_battery_data.xlsx

    # 2. Run inference and export predictions with a plot:
    python src/predict_excel.py sample_battery_data.xlsx --export predictions.xlsx --plot predictions.png

    # 3. Generate a sample Excel template:
    python src/predict_excel.py --generate-sample sample_battery_data.xlsx
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so the script can be run from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.custom_import import (
    parse_and_predict_custom_data,
    generate_sample_excel_template,
    export_predictions_excel,
    export_predictions_csv,
    PREDEFINED_DEFAULT_COLUMNS,
    OPTIONAL_COLUMNS,
)


def print_banner():
    print("=" * 72)
    print("  AMRITA DIGITAL TWIN - BATTERY SOC ESTIMATION (EXCEL TEST SUITE)  ")
    print("=" * 72)


def run_excel_test(file_path: str, export_path: str = None, plot_path: str = None):
    p = Path(file_path)
    if not p.is_file():
        print(f"[-] Error: File not found: '{file_path}'")
        sys.exit(1)

    print_banner()
    print(f"[*] Loading input file: {p.resolve()}")
    print(f"[*] File format: {p.suffix.upper()} ({p.stat().st_size / 1024:.1f} KB)")
    print(f"[*] Expected predefined columns: {', '.join(PREDEFINED_DEFAULT_COLUMNS)}")
    print(f"[*] Optional columns: {', '.join(OPTIONAL_COLUMNS)}")
    print("-" * 72)

    try:
        # Load and run forward inference with PureNumpyMLP
        result = parse_and_predict_custom_data(p, filename=p.name)
    except Exception as e:
        print(f"[-] Inference failed: {str(e)}")
        sys.exit(1)

    stats = result["stats"]
    df_proc = result["df_processed"]

    print("[+] Model Forward Inference Completed Successfully!")
    print("\n--- DATASET SUMMARY & TELEMETRY STATISTICS ---")
    print(f"  • Total Samples       : {stats['sample_count']} rows")
    print(f"  • Test Duration       : {stats['duration_s']} s ({stats['duration_min']} min)")
    print(f"  • Sampling Rate (dt)  : {stats['dt_s']} s")
    print(f"  • Voltage Range       : {stats['v_min']} V - {stats['v_max']} V (mean: {stats['v_mean']} V)")
    print(f"  • Current Range       : {stats['i_min']} A - {stats['i_max']} A (mean: {stats['i_mean']} A)")
    print(f"  • Initial Est. SOC    : {stats['soc_start']} %")
    print(f"  • Final Est. SOC      : {stats['soc_end']} %")
    print(f"  • Net SOC Depletion   : {stats['soc_delta']} %")

    # If ground truth SOC exists, print validation accuracy scorecard
    if stats.get("has_ground_truth"):
        print("\n--- GROUND TRUTH ACCURACY SCORECARD (TARGET: < 2.0% MAE) ---")
        mae = stats["ml_mae"]
        rmse = stats["ml_rmse"]
        max_err = stats["ml_max_error"]
        status = "PASSED (Within Target)" if mae < 3.0 else "WARNING (Exceeds Target)"
        print(f"  • Mean Absolute Error (MAE) : {mae:.2f} %  [{status}]")
        print(f"  • Root Mean Sq Error (RMSE) : {rmse:.2f} %")
        print(f"  • Maximum Absolute Error    : {max_err:.2f} %")
        print(f"  • Coulomb Counting MAE      : {stats.get('cc_mae', 'N/A')} %")

    print(f"\n--- SAMPLES TABLE: ALL {len(preview)} SAMPLES ---")
    preview = result["preview"]
    header = f"{'Row':<6}{'Time(s)':<10}{'Volt(V)':<10}{'Curr(A)':<10}{'V_smooth':<10}{'Pred_SOC(%)':<14}{'True_SOC(%)':<12}"
    print(header)
    print("-" * len(header))
    for row in preview:
        print(f"{row['row']:<6}{row['time_s']:<10.1f}{row['voltage']:<10.3f}{row['current']:<10.3f}{row['v_smooth']:<10.3f}{row['predicted_soc']:<14.2f}{str(row['true_soc']):<12}")

    # Determine export destination
    if not export_path:
        out_ext = ".xlsx" if p.suffix.lower() in [".xlsx", ".xls"] else ".csv"
        export_path = str(p.parent / f"predictions_{p.stem}{out_ext}")

    out_p = Path(export_path)
    print(f"\n[*] Exporting predictions to: {out_p.resolve()}")
    if out_p.suffix.lower() in [".xlsx", ".xls"]:
        excel_bytes = export_predictions_excel(df_proc)
        out_p.write_bytes(excel_bytes)
    else:
        csv_str = export_predictions_csv(df_proc)
        out_p.write_text(csv_str)
    print(f"[+] Predictions saved ({out_p.stat().st_size / 1024:.1f} KB).")

    # Optional plot generation
    if plot_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            t = df_proc["time_s"]
            
            # Subplot 1: Voltage
            ax1.plot(t, df_proc["V"], label="Terminal Voltage (V)", color="#2563eb", lw=1.5)
            ax1.plot(t, df_proc["V_smooth"], label="V_smooth (Causal MA)", color="#9333ea", lw=1.5, ls="--")
            ax1.set_ylabel("Voltage (V)")
            ax1.set_title(f"SOC Inference Profile - {p.name}")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper right")

            # Subplot 2: Current
            ax2.plot(t, df_proc["I"], label="Load Current (A)", color="#d97706", lw=1.2)
            ax2.set_ylabel("Current (A)")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="upper right")

            # Subplot 3: SOC
            ax3.plot(t, df_proc["soc_ml"], label="ML Predicted SOC (%)", color="#16a34a", lw=2)
            ax3.plot(t, df_proc["soc_cc"], label="Coulomb Counting (%)", color="#dc2626", lw=1.2, ls=":")
            if stats.get("has_ground_truth"):
                ax3.plot(t, df_proc["soc_true"], label="Reference True SOC (%)", color="#1f2937", lw=1.5, ls="--")
            ax3.set_ylabel("SOC (%)")
            ax3.set_xlabel("Time (seconds)")
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc="upper right")

            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"[+] Generated evaluation plot: {Path(plot_path).resolve()}")
        except Exception as e:
            print(f"[-] Plot generation failed: {e}")

    print("=" * 72)
    print("[+] Test completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Amrita BMS Digital Twin - Test battery SOC estimation using Excel (.xlsx/.xls) or CSV files."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to battery data file (.xlsx, .xls, or .csv)",
    )
    parser.add_argument(
        "--export",
        "-e",
        help="Path to export predictions (e.g. predictions.xlsx or predictions.csv)",
    )
    parser.add_argument(
        "--plot",
        "-p",
        help="Path to save evaluation plot image (e.g. results.png)",
    )
    parser.add_argument(
        "--generate-sample",
        "-g",
        help="Generate a ready-to-test sample Excel (.xlsx) file at the specified path",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=120,
        help="Number of synthetic samples when generating template (default: 120)",
    )

    args = parser.parse_args()

    if args.generate_sample:
        out_p = Path(args.generate_sample)
        print_banner()
        print(f"[*] Generating sample battery test profile: {out_p.resolve()}")
        raw_b = generate_sample_excel_template(n_samples=args.samples)
        out_p.write_bytes(raw_b)
        print(f"[+] Sample Excel template created successfully with {args.samples} rows ({len(raw_b)/1024:.1f} KB).")
        print(f"[*] Predefined columns included: {', '.join(PREDEFINED_DEFAULT_COLUMNS + OPTIONAL_COLUMNS)}")
        print(f"[*] Test it right away by running:\n    python src/predict_excel.py {args.generate_sample}")
        return

    if not args.file:
        parser.print_help()
        print("\n[!] Quick test: generate a sample template first:")
        print("    python src/predict_excel.py --generate-sample sample_battery_data.xlsx")
        print("    python src/predict_excel.py sample_battery_data.xlsx")
        sys.exit(1)

    run_excel_test(args.file, export_path=args.export, plot_path=args.plot)


if __name__ == "__main__":
    main()
