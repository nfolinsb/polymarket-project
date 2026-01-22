import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Monte Carlo probability function
# -------------------------------
def asset_touch_probs(
    S0,
    sigma,
    days,
    strikes,
    n_paths,
    r_pct
):
    T = days / 365
    n_steps = int(24 * days)
    dt = T / n_steps
    r = r_pct / 100

    Z = np.random.normal(0, 1, (n_paths, n_steps))
    increments = (
        (r - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * Z
    )

    log_paths = np.log(S0) + np.cumsum(increments, axis=1)
    paths = np.exp(log_paths)

    probs = []
    for K in strikes:
        if K >= S0:
            probs.append((paths.max(axis=1) >= K).mean())
        else:
            probs.append((paths.min(axis=1) <= K).mean())

    return probs


# -------------------------------
# Plot sample paths
# -------------------------------
def plot_sample_paths_with_strikes(
    S0,
    strikes,
    T,
    sigma,
    r_pct,
    n_paths=10
):
    n_steps = 200
    r = r_pct / 100
    strikes = np.asarray(strikes)
    dt = T / n_steps
    time = np.linspace(0, T, n_steps + 1)

    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = np.random.randn(n_paths, n_steps)
    log_paths = np.log(S0) + np.cumsum(drift + diffusion * Z, axis=1)
    paths = np.column_stack([np.full(n_paths, S0), np.exp(log_paths)])

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(n_paths):
        ax.plot(time, paths[i], lw=1.5, alpha=0.8)

    for K in strikes:
        ax.axhline(
            K,
            linestyle="--",
            alpha=0.6,
            color="red" if K > S0 else "blue"
        )

    ax.axhline(S0, color="black", lw=2, label="Spot")

    ax.set_title("Sample Paths with Strike Levels")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Asset Price")
    ax.grid(True)

    return fig


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Asset Touch Probability Simulator")

st.sidebar.header("Model Inputs")

S0 = st.sidebar.number_input("Spot Price (S₀)", value=88000.0, step=1000.0)
sigma = st.sidebar.number_input("Implied Volatility (%)", value=50, step=1)/100
r_pct = st.sidebar.number_input("Risk-Free Rate (%)", value=3.5, step=0.25)
days = st.sidebar.number_input("Days to Expiration", value=345, step=1)
n_paths = st.sidebar.number_input("Number of Simulated Paths", value=10_000, step=1_000)

strike_input = st.sidebar.text_input(
    "Strike Prices (comma-separated)",
    "70000, 80000, 110000, 120000, 150000"
)

# Parse strikes
strikes = np.array([float(x.strip()) for x in strike_input.split(",")])

T = days / 365

# -------------------------------
# Run simulation
# -------------------------------
if st.button("Run Simulation"):
    probs = asset_touch_probs(
        S0=S0,
        sigma=sigma,
        days=days,
        strikes=strikes,
        n_paths=int(n_paths),
        r_pct=r_pct
    )

    st.subheader("Touch Probabilities")
    for K, p in zip(strikes, probs):
        st.write(f"Strike {K:,.0f}: **{p:.2%}**")

    st.subheader("Sample Price Paths")
    fig = plot_sample_paths_with_strikes(
        S0=S0,
        strikes=strikes,
        T=T,
        sigma=sigma,
        r_pct=r_pct,
        n_paths=10
    )
    st.pyplot(fig)