"""
Preset: 3D Spiralgalaxie – Beispiel für N-Body-Initialisierung
"""
import math
import random

def generate_spiral_galaxy(num_stars=1000, arms=2, radius=50, spread=0.2, z_spread=1.5):
    # Mittlere Sternmasse (G=1); eingeschlossene Masse für Keplerian-Kreisbahngeschwindigkeit
    avg_mass    = (0.5 + 5.0) / 2.0          # erwartete Durchschnittsmasse
    M_total_est = num_stars * avg_mass        # Gesamtmasse-Schätzung (G = 1)

    stars = []
    for i in range(num_stars):
        arm = i % arms
        angle = (i / num_stars) * 2 * math.pi * arms + random.uniform(-spread, spread)
        r = random.uniform(0.1, 1.0) ** 0.5 * radius
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        z = random.gauss(0, z_spread)
        # Keplerian-Kreisbahngeschwindigkeit: v = sqrt(G * M_enc / r)
        # Eingeschlossene Masse für gleichförmige Scheibe: M_enc = M_total * (r/R)^2
        M_enc = M_total_est * (r / radius) ** 2
        v_kep = math.sqrt(M_enc / (r + 1e-6))   # G = 1
        vx = -(y / (r + 1e-6)) * v_kep
        vy =  (x / (r + 1e-6)) * v_kep
        vz = random.gauss(0, 0.02 * v_kep)
        mass = random.uniform(0.5, 5.0)
        stars.append({
            'position': (x, y, z),
            'velocity': (vx, vy, vz),
            'mass': mass
        })
    return stars

# Beispielaufruf
if __name__ == "__main__":
    stars = generate_spiral_galaxy()
    print(f"Generated {len(stars)} stars.")
