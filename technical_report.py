"""Generación del reporte técnico en texto plano."""

from __future__ import annotations

from typing import Any, Mapping

from ptc_model import SimulationResult


def build_technical_report(config: Mapping[str, Any]) -> str:
    g = config["geometry"]
    absorber = config["materials"]["absorber"]
    glass = config["materials"]["glass"]
    optics = config["optics"]
    environment = config["environment"]
    solar = config["solar"]
    operation = config["operation"]
    model = config["model"]
    solver = config["solver"]

    lines = [
        "============================================================",
        "REPORTE TECNICO - MODELO TERMICO TRANSITORIO DE PTC EN PYTHON",
        "============================================================",
        "",
        "FUENTES IMPLEMENTADAS",
        '1. "A_SOLAR_PARABOLIC_TROUGH_CONCENTRATOR_PT.pdf".',
        '2. "Análise e comparação do desempenho térmico de coletores solares planos e parabólicos no TRNSYS.pdf".',
        "3. Forristall, R. (2003), Heat Transfer Analysis and Modeling of a Parabolic Trough Solar Receiver Implemented in Engineering Equation Solver, NREL/TP-550-34169.",
        "",
        "ESTRUCTURA DEL MODELO",
        f"- Red nodal radial: HTF -> absorbedor -> cubierta de vidrio -> ambiente/cielo.",
        f"- Discretizacion axial: {int(g['Nseg'])} volumenes de control.",
        "- Estados por nodo: temperatura del HTF, del absorbedor y de la cubierta.",
        "- Propiedades rho(T), mu(T), Cp(T) y k(T) actualizadas localmente.",
        "- Integrador rigido: scipy.integrate.solve_ivp con metodo BDF, equivalente funcional a ode15s.",
        "",
        "GEOMETRIA",
        f"L = {g['L']:.6g} m",
        f"W = {g['W']:.6g} m",
        f"f = {g['f']:.6g} m",
        f"D2 = {g['D2']:.6g} m; D3 = {g['D3']:.6g} m; D4 = {g['D4']:.6g} m; D5 = {g['D5']:.6g} m",
        "",
        "MATERIALES Y OPTICA",
        f"Absorbedor: rho = {absorber['rho']:.6g} kg/m3; Cp = {absorber['Cp']:.6g} J/(kg K); k = {absorber['k']:.6g} W/(m K); eps = {absorber['eps']:.6g}; alpha = {absorber['alpha']:.6g}",
        f"Vidrio: rho = {glass['rho']:.6g} kg/m3; Cp = {glass['Cp']:.6g} J/(kg K); k = {glass['k']:.6g} W/(m K); eps = {glass['eps']:.6g}; tau = {glass['tau']:.6g}; alpha = {glass['alpha']:.6g}",
        f"Reflectividad = {optics['reflectivity']:.6g}; factor de interceptacion = {optics['intercept_factor']:.6g}; suciedad = {optics['dirt_factor']:.6g}; sombra = {optics['shade_factor']:.6g}",
        "",
        "AMBIENTE, OPERACION Y SOLVER",
        f"Tamb = {environment['Tamb_K'] - 273.15:.6g} C; Tsky = Tamb - {environment['sky_delta_K']:.6g} K",
        f"Viento = {environment['wind_m_s']:.6g} m/s; presion = {environment['pressure_Pa']:.6g} Pa",
        f"Fluido = {operation['fluid']}; mdot = {operation['mdot']:.6g} kg/s; Tin = {operation['Tin_K'] - 273.15:.6g} C",
        f"Intervalo = {operation['t_start_s']/3600:.6g} a {operation['t_end_s']/3600:.6g} h; salida cada {operation['output_step_s']:.6g} s",
        f"rtol = {solver['rtol']:.3e}; atol = {solver['atol']:.3e}; max_step = {solver['max_step_s']:.6g} s",
        f"Cubierta de vidrio = {bool(model['has_glass'])}; modelo anular = {model['annulus']}; correlacion interna = {model['internal_correlation']}",
        "",
        "MODELO SOLAR",
        f"Modo = {solar['mode']}",
        f"Parishwad: A = {solar['A']:.6g} W/m2; B = {solar['B']:.6g}; dia = {int(solar['day_of_year'])}; latitud = {solar['latitude_deg']:.6g} deg",
        f"Constante: DNI = {solar['DNI_constant_W_m2']:.6g} W/m2; theta = {solar['angle_constant_deg']:.6g} deg",
        "",
        "ECUACIONES RESUELTAS POR NODO i",
        "Balance HTF:",
        "C_f,i*dTf_i/dt = mdot*Cp_up*(T_up - Tf_i) + Q_absorbedor_a_HTF_i",
        "Balance absorbedor con vidrio:",
        "C_abs,i*dTabs_i/dt = Qsolar_abs_i - Q_absorbedor_a_HTF_i - Qrad_abs_vid_i - Qconv_anular_i - Qsoportes_i",
        "Balance cubierta:",
        "C_vid,i*dTvid_i/dt = Qsolar_vid_i + Qrad_abs_vid_i + Qconv_anular_i - Qconv_vid_amb_i - Qrad_vid_cielo_i",
        "Balance absorbedor sin vidrio:",
        "C_abs,i*dTabs_i/dt = Qsolar_abs_i - Q_absorbedor_a_HTF_i - Qconv_abs_amb_i - Qrad_abs_cielo_i - Qsoportes_i",
        "",
        "CONVECCION INTERNA",
        "Re = 4*mdot/(pi*D2*mu)",
        "Pr = Cp*mu/k",
        "Laminar: Nu = 4.36",
        "Dittus-Boelter: Nu = 0.023*Re^0.8*Pr^0.4",
        "Gnielinski-Forristall:",
        "f = (1.82*log10(Re) - 1.64)^(-2)",
        "Nu = (f/8)*(Re-1000)*Pr/[1 + 12.7*(f/8)^0.5*(Pr^(2/3)-1)]*(Pr_bulk/Pr_wall)^0.11",
        "h_int = Nu*k/D2",
        "",
        "RED DE RESISTENCIAS",
        "R_int = 1/(h_int*pi*D2*dx) + ln(D3/D2)/(4*pi*k_abs*dx)",
        "Q_absorbedor_a_HTF = (Tabs - Tf)/R_int",
        "Qrad_abs_vid = sigma*pi*D3*dx*(Tabs^4-Tvid^4)/[1/eps_abs + (D3/D4)*(1/eps_vid-1)]",
        "Qconv_anular = h_anular*pi*D3*dx*(Tabs-Tvid)",
        "Qconv_ext = h_ext*pi*Dext*dx*(Tsuperficie-Tamb)",
        "Qrad_cielo = eps*sigma*pi*Dext*dx*(Tsuperficie^4-Tsky^4)",
        "",
        "RADIACION Y OPTICA",
        "DNI = A*exp(-B/cos(z))",
        "cos(z) = sin(phi)*sin(delta) + cos(phi)*cos(delta)*cos(omega)",
        "delta = 23.45*sin[360*(284+n)/365]",
        "omega = 15*(12-LAT)",
        "IAM = 1 + 0.000884*theta/cos(theta) - 0.00005369*theta^2/cos(theta)",
        "EndLoss = 1 - f*tan(theta)/L",
        "",
        "RESULTADOS GLOBALES",
        "Q_util = mdot*Cp_medio*(Tout-Tin)",
        "Q_perdidas = sum(Qconv_ext + Qrad_cielo + Qsoportes)",
        "U_L = Q_perdidas/[W*L*(Tabs_media-Tamb)]",
        "eta = 100*Q_util/Q_incidente_apertura",
    ]
    return "\n".join(lines)


def result_summary(result: SimulationResult) -> str:
    dni = result.scalar_diag["DNI_W_m2"]
    k = int(dni.argmax())
    cfg = result.config
    lines = [
        f"Escenario: {cfg['operation'].get('name', 'sin nombre')}",
        f"Solver: BDF; {result.solver_message}",
        f"Evaluaciones RHS: {result.nfev}; evaluaciones Jacobiano: {result.njev}; factorizaciones LU: {result.nlu}",
        f"Pico DNI: {dni[k]:.3f} W/m2 a LAT {result.LAT_h[k]:.3f} h",
        f"Temperatura de salida: {result.Tout_C[k]:.3f} C",
        f"Temperatura media del absorbedor: {result.Tabs_mean_C[k]:.3f} C",
        f"Temperatura media del vidrio: {result.Tglass_mean_C[k]:.3f} C",
        f"Q_util: {result.scalar_diag['Quseful_W'][k]:.3f} W",
        f"Q_perdidas: {result.scalar_diag['Qloss_W'][k]:.3f} W",
        f"Eficiencia: {result.scalar_diag['eta_pct'][k]:.3f} %",
        f"U_L: {result.scalar_diag['UL_W_m2K'][k]:.6f} W/(m2 K)",
        f"Re interno a la salida: {result.node_diag['Re_internal'][k, -1]:.3f}",
        f"h_int a la salida: {result.node_diag['h_internal_W_m2K'][k, -1]:.3f} W/(m2 K)",
    ]
    return "\n".join(lines)
