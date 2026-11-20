"""Validación con series de termopares, independiente de los presets documentales.

ÚNICAS observaciones usadas: DATE, TIME, termopar_1 (Tin), termopar_2
(Tout). La radiación, caudal y demás columnas de origen se descartan.
La irradiancia es una PREDICCIÓN de cielo claro; no es una medición.
"""

from __future__ import annotations

from copy import deepcopy
import base64
from datetime import datetime, date, time
from io import BytesIO
import zlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ptc_model import PTCSimulator

BUNDLED_FOZ_CSV = Path(__file__).resolve().parent / "data" / "foz_2026_11_16_termopares.csv"

# Copia autónoma de los cuatro campos autorizados del ensayo de 16/11/2026.
# Streamlit Cloud puede desplegar módulos Python sin incluir el directorio data/.
# Se usa ÚNICAMENTE cuando falta el CSV distribuido con el proyecto.
# Generado desde data/foz_2026_11_16_termopares.csv (zlib + base85).
# No contiene valores del sensor de radiación ni otras variables de la planilla.
_EMBEDDED_FOZ_CSV_B85 = (
    "c-nQFvCd_;4TSpud6OU4k|<Krc7fv}MLPFJij+YDBkzxBhCI8@B`;tsu<(ZwMNyPS_v~N)^6M{Ozy9s-zkL1opTGa>-"
    "~ai?@4xlme*W>7@!QXT{!_P~djDy(VPCy{YxCROU$ec$zI46hBtXA)0ye(XUb+eZ3jNhCFkfdRz^Kq)vwf+&<|zPISQi*CZ3$RZxc_T@X@Bie0"
    "8o&ZH2|7{O@+y6w%2I|FbQCY@#=Y+ClvHfbHADvz*Ug_Ivttp3BZ3BkURR9=xg-?fWjSO0q7}Fy;EVn_5`3(0IuLe%ruKWsGzU*NnwP-BBnC|5"
    "(-`L=UD|X3D^bfCOiQ%6xIYVv;bUT4=UssG!zaMrWODc?A58DIR@4wfQ3I-D*#s*1a^*Lg@W8^u2<^?;0me$Mjm2^f-"
    "L~WEOsdDj=Ji3u~?HB{Z8`~W#7Ff@4hCGOK!AK*atv61>jXZ?E~O7d3Qv|2FwLBdMJ#8Ei7r8V}!!po3{zrDgd8_Sv}1mF^hAcvt|K!4AP;{cL"
    "HX_pp8SE0&oRmGtAq6Xeby{y-xwS!g7VQn!`fjD1hDzAUezhFtUQz<Uu_^VHLn6(5WCf4?7fQ2e7mNT){dPs|fOhf+-J$RRFHA7J!ieX>sB-VL"
    "eCz_?guYfZWWbheGENdHYL$PQAf%CrwjEC=4D<S^=TJJmf|y5(?r9y#QR{o;a9AS__$>uz0lfQ2?%R97ts^&)I#r^~?&uPu}i89^yH>rE&9~0&"
    "oT7VcfDwz<N%9`w+tlg?s8<;ADjz3PxZ{0l30F_3A{OxIeT5VC({z1nyq}q?(*96vhOQd-ZIgFqQX^6y)ro;1`u#a<hkmafp*vbB<8JH0Q+%z{"
    "T55l-"
    "^JiAfd1nKpO=Rv!FumQFA`CFalNpF^fYWt>)Bc7MnmT0G|ccViwCiy(Wd{#RaqgT;blt>RGI~%b~;EYf>wKP+*Yx^q<dtvfs6^YMRe|5~tq1B;"
    "+BU`y?`1T`=Yeg?a1Yp5Un_wLN#*1t1^p+8%|Qhh4N{?SF`27hO(!C@cYxdz6k)Kt-M6!l@$^wgAXSvq~tOL0}ZX<SL*?-"
    "Genx(`!=Mqb{%uz!ii;<nux`6p)8|a_s_e1!D~Px~LWk0jvTD1)pYWq`LlJK?}eYa2(uOq^+kr6t=?S<l{iSCJm?m1#*{DuSr9}Kw%evD;S4(Z"
    ">0TOYoV|X02u`Ug?nY?V!fB#1$ro)#UVxkcnq6XZe@k>#IRuo-|v$ITtPei=1hP*c`!KE_e(E<P?!Kl24*NAQF22~fZnS$^3X4E7hMZL2?&Ar-"
    "QD4uUExM@fwK$173P6+x2&+Ja65xhK?=YXkV#8kufsJDdiHxP>Z#UZQvu>kR8}~Q0SyqZ2XhpF$8h%rUJp``G%0Lf2g{uchMf?_*17d|TJHPbA"
    "mH0yPXTxg%6-w&YF^juvp_al>JJ3ePi@v@-WS(34+5ie7N-DQLAZHyqONOpg&X9Zg;)W&0&ashU8!KY*QCaYlb5?|tfuk2=y@&dQ78`-"
    "0<O^fK*>R_P{1wdq>B~-^)uJIeNNPlwGj5G)(gNDR;x~&4%Zwt>7BM0fY*aJQcVh@#oiq3QF|AFD-"
    "6~{Pk9(U6z<`Uf>{L+HOV(IO=~emDBQCfk7iZ?>f4`NxTj1GiJgTYM^TeQLO}`S#TxV3n~T6H09P<7<Tf}o6gJ=Gb}xX~-"
    "JQKUPt#&|3tMsS55q#is33WW6$;mYTtHg^{83@$#oCVoo7g!8;0mjw4HmOFp>PNcEdXtP?z535Kv<JFln;SZ0Ipzs$j!&1^iV)E$DObMl(lp?M"
    "DDc0n#3whoH%V4fI1)ma#f3j0;Xx{4TOLzw2f!jnO0Ngqj30;YXuN7ECQ)uM4wqS3~^ckxWXi`vx0>Jj^<UgAhA$5pA=Y=H_2TXLj`E_QX!BNw"
    "L?Mppq(p9b|~OBxQi|)Clu5G*en27Sj=R;iA}Fb#X}8%T>whJhnQ)I*+XFw$lZQ=O&TvbJ9)hh*E|T|U|m&^BNUVm`}tU$5(;Qy_gHKN;0pM8V"
    "yCCz4%h4o%9<Rr078L@%1vx)D4<X7;$}WI6gG#*_lIeZf^aQb0l30Z)<XuoCV98l>Vh%7CLbW*Y3B}wE*}bJhk`JMy!GY@1r+TmcK1A?fMf9#n"
    "bZ~vn?ODew1vXqr8nn6`=c-joC0u#-2hmQVT1x2DNc<7Q0A~R$ebwkn%t3xT>-"
    "9e?+@0BR;)7=u9drh%mQ$Qy#VCBsTvAr1JG3f+I*P+<n6CslLs$a7uZ?=uF$EFO^~iokp5*oy$up@h5P3PS{`DDf)U6&OLr(BQ9ZqZbhu_$I0m"
    "4Y=6PbE!h5$LaLulORObh60*IeK<6hwXf-&Hl2Z4KaH%ZG)z!k3VwwdxU;F<?BfK<=4c?oz9@7^>`C+~o3b_Ll0_9*~YxYq--"
    "<Q4~PwJY4enk9ch;71IWQa!?VV-"
    "+~WsQ_2NT4c{V!gq5OzU;QLSpcEX0rb3@>xrQOkX`_;a1Rt)>9>Osz8e5zcz<mk;k!{wj6w4>Pbe%OLJJ^bI2<A=xbMag0^sxlfWp8VsjVWD?z"
    ";`n;>&h=ryb$DZE%`9Ig}B;`vTX>0T>0~3S$t+F-"
    "R!f)8F#3IKp>dU=qk{;lBF<o4{HH00qHQ&jof%fY+p;R|A*@;0ow4r%IHCf)VJw09@f7Np_08>AssAAAq$2Py*Qg`2ll;?}h?n$W6?BxAD!~PB"
    "qQ*C<uTf1)xumtEHRy2;Y5y`{yUX70`lC(I?$^UqG3MqXpm!Q#eH07w)?OFvwhT?z<5KfxP|QcLO*CP67N;A=jk)Zp6Sm<lXJQ`vUhqSpymhi_"
    "^><#(nn%HWgM*wLJ<Oz^Q;xm|v@>UxG&X?t4;PBer_F6_4=UM*FaO`m|t#?}mZ^$PL7O_XTuNArEmvVQv7r3Lq2+<V3meHoVXRI933zU`ikb>A"
    "o97aGJT1y6-"
    "l=%dJryN$$HZum*uq08lWP;)S0<6W}#ztf&k00&s=9vo4TZko#_&+KeIhYWLmWfsr$*p<o?=&H`|S#TYErgZpj(6ToT(5HTzQd59GX_v*f+XvO"
    "ZkjjxLr7&(Uh%wiK51rQ1cKrUMM-H1T}NGkwWI2=M!!MN`>`YZsm0JQnt<AXvE1)s&V0tf{IpeacA-NtW|09XOIf>6Oy9^7|dU@%c<7C<O)7J1"
    "{i@3zHWEk(C?-)$?FJ6{*wck_A!U{`=EtQ*5j4-"
    "_MOHvm_d1>g#%AEvoN0YT36@;}0NN075xZm&sxqqSyHwC=l&@3a8&TAWabA*=wD=YI1cQVj09p}<+>dT`&3D;5DunRMR`U;xPe!hQGmA@XXv??"
    "#Xp(4xz^?~Z9|H46y^G*T^6?Y<jVtPNnM2I9UOcaH%epE&NjjUP~5Am7a0cVmbNAUiep-A2C&AhjU(-MA=HVM_tHf-q4h?QZwo$fPTr0tkiMf-"
    "HTUFv52u$U#60AZD>X0OwH<2007B6(-"
    "l>{Z1?4yAdQ4_5K@*JieRWf93B$<ni6~{)piefbx<yHWYeRU`^tR<<p!6;0l|G>UjtWh0_6y3V=e#oB6WT%q4vH;y2#Q(uGsPcjJoH0KES+AmO"
    "_aBxkW_0r)H?Gr4jM77CrSc(+j!zI*W(kUEAH3b-"
    "~~>eUjy8>!|jPM+p|)}ntHVuykk?fb7V622RN0gzn)u5eC0g!^t}lCwCw0Ae+79#(oqk?`FYm;=E3kI@po8&_@z(ptFh#t?@?NC8Amb^<Ar?z<"
    "5K73L`bj{%8VsSQf_ZYU^#r3DZQoQ38@X()6CIadMvSc{dso`mm49(*-h0sL4CP4{XE-"
    ";EfQ$Koh}AA3{MW3q(rjv;3H^+CdSV=dSQi&;E1xmZ!faz8aGtf)N1cxn<pD?b)V_-"
    "^d(JBwX3G56ik<?>sQgzv`2SzOJu+~a$n93NsBPmiL4<rtot6fU=E0l0#!55QBC;w3i<K)*NAmEO!Hd^a4oC+ZYHWO4woycSQNgc;oF<dyK<ix"
    "0Ug|3o6;yJK&j^qGf*?_T`q7enM!KQ)Oz^A9}|z8k>dEOI7KDCqhy%@YcXG2}zdefQ$GNh|-RCgHp3b_t{)<>`}$K+c2vZWwA#GdFYh-"
    "L%8?EZlcrKsdz83K9y!MD0-kt^g~yi~iug8+LenP?(|MKtA={cLU%w^NHiWn~s|ml~&VzH)hd4#9*O-"
    "=Y>;jfA`%H!znH~?z_JO#d7aZXsqgdEV}Oo&<Uh{;l3NdBG3vTDr(e2%%>*%1|V7hp}-+#8p3^dRMbiSLc(`LfkVucYWLlEyO;pf3P1^T#-"
    "Mo!35CH1BE0}yK}MY@_uT-rUM%<Bd{|%%SwTaga|l}n;CVRnYq#{eDB-"
    ")22Lq7rF!QO&dxzPrO4NE{Xasfv#4I*|+#lR`<5*t+a#g$U#_NXzc{NWc?DJt3=TYD+^4;EjH)3FrCI#RMi<!*b-"
    "hDToCgf}P;(Fk|o6jQva;J6Qjpva`U=@H8xR<@%i|eBM?!~WJcK#er!gnveneSfQAKZ6eU@?YV_U^m!ZeSl2G!%TA`EYmN&8G<f`Qq)q8$hFi7"
    "J$-mHv%gUu^t7^BA@>5yBGicCV-"
    "r%9SYYi?)+9P;kyyT0?_kfc}?<rdjPWlTw(eUZ#6l?cf)Zrlkd+9GkiBZJ6F>(&_m%c)$iX8&c}E2b9xLh5-"
    ">u+4uf0;p!A82(|m7Y^YPvMRCVWn`8Xfnjei3RVC5KQD0HTJ&jN@TsIYU8`Y7C@mGnR{AK%TN?TY>|AKyKBJ+QQz^YPurUqB?CSu=e1c87ASdK"
    "bV|*!-!;@{u&dcVn3rpL16MxPl3O<|wAgn%uk|;CIprP!4fZZKH_6eK(qLtG;HLkMG`y0iTPc<qqGy;-"
    "kJZ3&1mJ{LZSUAj5YXVo+r!{|C1C1=a"
)
# Foz do Iguaçu. Coordenadas aproximadas del centro urbano, no del lugar del sensor.
FOZ_LATITUDE = -25.516
FOZ_LONGITUDE = -54.585
FOZ_UTC_OFFSET = -3.0


def _parse_datetime(date_value: Any, time_value: Any) -> datetime:
    if isinstance(date_value, datetime):
        d = date_value.date()
    elif isinstance(date_value, date):
        d = date_value
    else:
        raw_date = str(date_value).strip().split(" ")[0]
        d = None
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                d = datetime.strptime(raw_date, fmt).date()
                break
            except ValueError:
                continue
        if d is None:
            raise ValueError(f"Fecha no reconocida: {date_value!r}")

    if isinstance(time_value, datetime):
        t = time_value.time()
    elif isinstance(time_value, time):
        t = time_value
    else:
        raw_time = str(time_value).strip()
        t = None
        for fmt in ("%H:%M:%S", "%H:%M", "%H:%M:%S.%f"):
            try:
                t = datetime.strptime(raw_time, fmt).time()
                break
            except ValueError:
                continue
        if t is None:
            raise ValueError(f"Hora no reconocida: {time_value!r}")
    return datetime.combine(d, t)


def _temperature(value: Any, title: str) -> float:
    try:
        out = float(str(value).strip().replace(",", "."))
    except (ValueError, TypeError):
        raise ValueError(f"Lectura no numérica en {title}: {value!r}") from None
    if not np.isfinite(out):
        raise ValueError(f"Lectura no finita en {title}: {value!r}")
    return out


def normalise_measurements(raw: pd.DataFrame) -> pd.DataFrame:
    """Conserva únicamente los instantes y las dos temperaturas identificadas.

    La columna mal escrita `temopar_1` del XLSX original está vacía y NO
    sustituye a `termopar_1`. Se admiten encabezados canónicos del CSV limpio.
    """
    labels = {str(col).strip().casefold(): col for col in raw.columns}
    aliases = {
        "date": ("date", "data"),
        "time": ("time", "hora"),
        "tin": ("termopar_1_c", "termopar_1"),
        "tout": ("termopar_2_c", "termopar_2"),
    }
    found = {}
    for role, choices in aliases.items():
        key = next((labels[c] for c in choices if c in labels), None)
        if key is None:
            raise ValueError(f"Falta columna para {role}: {', '.join(choices)}")
        found[role] = key

    points: list[tuple[datetime, float, float]] = []
    for index, row in raw.iterrows():
        fields = [row[found[k]] for k in ("date", "time", "tin", "tout")]
        if all(pd.isna(item) or str(item).strip() == "" for item in fields):
            continue
        if any(pd.isna(item) for item in fields):
            raise ValueError(f"DATE/TIME/T1/T2 incompletos en la fila {index + 2}")
        moment = _parse_datetime(fields[0], fields[1])
        points.append((moment, _temperature(fields[2], "termopar_1"), _temperature(fields[3], "termopar_2")))
    if len(points) < 2:
        raise ValueError("Se necesitan al menos dos registros con fecha, hora, T1 y T2.")
    points.sort(key=lambda item: item[0])
    if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
        raise ValueError("Hay instantes de medición repetidos; revise las fechas y horas.")
    base_day = datetime.combine(points[0][0].date(), time.min)
    return pd.DataFrame({
        "datetime_local": [p[0] for p in points],
        "DATE": [p[0].strftime("%d/%m/%Y") for p in points],
        "TIME": [p[0].strftime("%H:%M:%S") for p in points],
        "t_s": [(p[0] - base_day).total_seconds() for p in points],
        "Tin_exp_C": [p[1] for p in points],
        "Tout_exp_C": [p[2] for p in points],
    })


def load_bundled_measurements() -> pd.DataFrame:
    """Carga T1/T2 del CSV incluido o de la copia embebida si falta en Cloud.

    La copia embebida evita FileNotFoundError cuando GitHub/Streamlit Cloud
    despliega el código sin `data/`. No consulta servidores externos, no altera
    la campaña y nunca lee la columna de radiación del archivo original.
    """
    if BUNDLED_FOZ_CSV.is_file():
        raw = pd.read_csv(BUNDLED_FOZ_CSV, sep=",", encoding="utf-8-sig", dtype=str)
    else:
        csv_bytes = zlib.decompress(base64.b85decode(_EMBEDDED_FOZ_CSV_B85))
        raw = pd.read_csv(BytesIO(csv_bytes), sep=",", encoding="utf-8-sig", dtype=str)
    return normalise_measurements(raw)


def load_measurements_upload(payload: bytes, filename: str) -> pd.DataFrame:
    """Importa un CSV/XLSX futuro sin usar radiación ni columnas adicionales."""
    filename = str(filename).lower()
    if filename.endswith(".xlsx"):
        # Dependencia openpyxl ya declarada por la aplicación para importar XLSX.
        raw = pd.read_excel(BytesIO(payload), dtype=str)
    elif filename.endswith(".csv"):
        raw = pd.read_csv(BytesIO(payload), sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    else:
        raise ValueError("Formato no soportado; utilice CSV o XLSX.")
    return normalise_measurements(raw)


def apply_measurements(config: Mapping[str, Any], measurements: pd.DataFrame, *, source: str) -> dict[str, Any]:
    """Añade condiciones de borde T1(t) y referencias T2(t), sin calibrar parámetros."""
    df = measurements.copy()
    if df.empty:
        raise ValueError("La serie experimental está vacía.")
    cfg = deepcopy(dict(config))
    times = df["t_s"].astype(float).tolist()
    inlet = df["Tin_exp_C"].astype(float).tolist()
    outlet = df["Tout_exp_C"].astype(float).tolist()
    dates = df["DATE"].tolist()
    hours = df["TIME"].tolist()
    first_date = df.iloc[0]["datetime_local"].date()

    op = cfg["operation"]
    op["Tin_K"] = inlet[0] + 273.15
    op["Tin_profile"] = {"t_s": times, "Tin_K": [v + 273.15 for v in inlet]}
    op["output_times_s"] = times
    op["t_start_s"], op["t_end_s"] = times[0], times[-1]
    op["output_step_s"] = float(np.median(np.diff(times)))
    op["name"] = f"Validación experimental · {first_date.isoformat()}"

    solar = cfg["solar"]
    solar["date_origin"] = first_date.isoformat()
    solar["day_of_year"] = first_date.timetuple().tm_yday
    cfg["experimental_validation"] = {
        "source": source,
        "columns_used": ["DATE", "TIME", "termopar_1", "termopar_2"],
        "DATE": dates,
        "TIME": hours,
        "t_s": times,
        "Tin_exp_C": inlet,
        "Tout_exp_C": outlet,
        "timezone": "America/Sao_Paulo (UTC−03)",
    }
    cfg["preset_meta"] = {
        **cfg.get("preset_meta", {}),
        "family": "Experimental Foz termopares",
        "source": source,
        "date_label": f"{first_date.isoformat()} · {hours[0]}–{hours[-1]}",
        "city": "Foz do Iguaçu, PR, Brasil",
        "assumptions": [
            "T1(t) es la condición de entrada; T2(t) es exclusivamente la referencia de salida.",
            "Caudal, geometría, viento y temperatura ambiente son parámetros editables; no provienen de esta planilla.",
            "La radiación se calcula con cielo claro y geometría solar, sin leer el sensor del XLSX.",
            "La condición térmica inicial de los sólidos es supuesta; no se ajusta con T2.",
        ],
    }
    return cfg


def run_experimental_validation(config: Mapping[str, Any], fluid_database: Mapping[str, Any]) -> dict[str, Any]:
    """Predice T2 en cada instante medido y calcula errores sin ajuste de parámetros."""
    raw = config.get("experimental_validation")
    if not isinstance(raw, dict):
        raise ValueError("La configuración activa no contiene una serie de termopares.")
    result = PTCSimulator(config, fluid_database).simulate()
    times = np.asarray(raw["t_s"], dtype=float)
    if len(result.t_s) != len(times) or not np.allclose(result.t_s, times, atol=1e-6, rtol=0):
        raise RuntimeError("La salida del solver no coincide con la malla temporal experimental.")
    ref = np.asarray(raw["Tout_exp_C"], dtype=float)
    pred = np.asarray(result.Tout_C, dtype=float)
    tin = np.asarray(raw["Tin_exp_C"], dtype=float)
    resid = pred - ref
    metrics = {
        "n": int(len(times)),
        "RMSE_Tout_C": float(np.sqrt(np.mean(resid**2))),
        "MAE_Tout_C": float(np.mean(np.abs(resid))),
        "Bias_Tout_C": float(np.mean(resid)),
        "Max_abs_error_C": float(np.max(np.abs(resid))),
        "n_Tout_below_Tin": int(np.sum(ref < tin)),
    }
    reference = pd.DataFrame({
        "DATE": raw["DATE"],
        "TIME": raw["TIME"],
        "Tin_exp_C": tin,
        "Tout_exp_C": ref,
        "Tout_model_C": pred,
        "Error_model_minus_exp_C": resid,
        "DNI_synthetic_W_m2": result.scalar_diag["DNI_W_m2"],
        "Cos_theta_model": result.scalar_diag["cosTheta"],
        "G_aperture_model_W_m2": result.scalar_diag["G_aperture_W_m2"],
    })
    return {"table": reference, "result": result, "metrics": metrics,
            "note": "Validación comparativa de temperatura de salida; radiación de cielo claro estimada; caudal y ambiente editables, no medidos en la serie utilizada."}
