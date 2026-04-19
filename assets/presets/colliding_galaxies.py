"""
Preset: Zwei kollidierende Spiralgalaxien
"""
from assets.presets.spiral_galaxy import generate_spiral_galaxy
import numpy as np

def generate_colliding_galaxies(stars_per_galaxy=25000):
    """
    Zwei Spiralgalaxien auf Kollisionskurs (Off-Center für Gezeiteneffekte).
    Galaxie 1: links, bewegt sich nach rechts-oben
    Galaxie 2: rechts, bewegt sich nach links-unten
    """
    offset = 220.0     # Abstand zwischen Zentren (erheblich größer für realistische Anflugphase)
    approach = 0.6     # Annäherungsgeschwindigkeit (Gravitation beschleunigt die Galaxien)
    transverse = 0.15  # Queranteil für Off-Center-Kollision

    # Galaxie 1 – gelb
    g1 = generate_spiral_galaxy(num_stars=stars_per_galaxy, arms=2, radius=35)
    for s in g1:
        p = np.array(s['position']); v = np.array(s['velocity'])
        p[0] -= offset
        v[0] += approach
        v[1] += transverse
        s['position'] = tuple(p); s['velocity'] = tuple(v)

    # Galaxie 2 – cyan, entgegengesetzte Rotation (retrograd) für dramatischere Kollision
    g2 = generate_spiral_galaxy(num_stars=stars_per_galaxy, arms=3, radius=30)
    for s in g2:
        p = np.array(s['position']); v = np.array(s['velocity'])
        p[0] += offset
        v[0] -= approach
        v[1] -= transverse
        v[1] = -v[1]  # umgekehrte Rotation
        s['position'] = tuple(p); s['velocity'] = tuple(v)

    return [g1, g2]


if __name__ == "__main__":
    galaxies = generate_colliding_galaxies()
    print(f"Galaxies: {[len(g) for g in galaxies]}")
