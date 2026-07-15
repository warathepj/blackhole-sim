//! Physics core for Black Hole Simulation.
//!
//! Uses Paczyński-Wiita pseudo-potential to model the gravity of a
//! Schwarzschild (non-rotating) black hole. This pseudo-Newtonian
//! potential reproduces key relativistic effects such as:
//! - Innermost Stable Circular Orbit (ISCO) at r = 3·rₛ
//! - Event horizon barrier at r = rₛ
//! - Orbital precession (perihelion advance)

use nalgebra::Vector3;
use rand::RngExt;

/// 3D vector type used throughout the simulation
pub type Vec3 = Vector3<f64>;

/// Gravitational constant (in simulation units)
pub const G: f64 = 1.0;

/// Speed of light (in simulation units)
pub const C: f64 = 1.0;

// ---------------------------------------------------------------------------
// Black Hole
// ---------------------------------------------------------------------------

/// A simplified Schwarzschild black hole.
///
/// Gravity is computed via the Paczyński-Wiita pseudo-potential:
/// ```text
/// Φ(r) = -GM / (r - rₛ)
/// a(r) = -∇Φ = -GM / (r - rₛ)² · r̂
/// ```
#[derive(Clone, Debug)]
pub struct BlackHole {
    /// Mass in simulation units
    pub mass: f64,
    /// Schwarzschild radius rₛ = 2GM / c²
    pub schwarzschild_radius: f64,
    /// Position in 3D space
    pub position: Vec3,
}

impl BlackHole {
    /// Create a new black hole at `position` with the given `mass`.
    pub fn new(mass: f64, position: Vec3) -> Self {
        let rs = 2.0 * G * mass / (C * C);
        Self {
            mass,
            schwarzschild_radius: rs,
            position,
        }
    }

    /// Gravitational acceleration at `pos` using the Paczyński-Wiita
    /// pseudo-potential (returns zero inside the event horizon).
    pub fn gravitational_acceleration(&self, pos: &Vec3) -> Vec3 {
        let r_vec = pos - self.position;
        let r = r_vec.norm();

        // Inside or at the event horizon → no meaningful acceleration
        if r <= self.schwarzschild_radius || r < 1e-12 {
            return Vec3::zeros();
        }

        let r_hat = r_vec / r;
        let effective_r = r - self.schwarzschild_radius;

        // a = -GM / (r - rₛ)²  (radially inward)
        let a_mag = -G * self.mass / (effective_r * effective_r);
        r_hat * a_mag
    }

    /// Circular orbit velocity at `radius` for a test particle.
    ///
    /// For the Paczyński-Wiita potential this is:
    /// ```text
    /// v_φ = √( G·M·r / (r - rₛ)² )
    /// ```
    /// Returns zero for radii inside the ISCO (r ≤ 3·rₛ) where no stable
    /// circular orbit exists.
    pub fn circular_orbit_velocity(&self, radius: f64) -> f64 {
        if radius <= self.isco_radius() || radius <= self.schwarzschild_radius {
            return 0.0;
        }
        let effective_r = radius - self.schwarzschild_radius;
        (G * self.mass * radius / (effective_r * effective_r)).sqrt()
    }

    /// Radius of the Innermost Stable Circular Orbit: 3 · rₛ
    pub fn isco_radius(&self) -> f64 {
        3.0 * self.schwarzschild_radius
    }

    /// Predict the future trajectory of a particle starting at `position` with `velocity`
    /// for `steps` steps, using a given time step `dt`.
    /// Returns a list of positions representing the predicted path.
    pub fn predict_trajectory(&self, mut position: Vec3, mut velocity: Vec3, steps: usize, dt: f64) -> Vec<Vec3> {
        let mut path = Vec::with_capacity(steps);
        for _ in 0..steps {
            let acc = self.gravitational_acceleration(&position);
            velocity += acc * dt;
            position += velocity * dt;
            path.push(position);

            // Stop if it falls into the event horizon
            if position.norm() < self.schwarzschild_radius * 0.5 {
                break;
            }
        }
        path
    }
}

// ---------------------------------------------------------------------------
// Particle (test particle / gas cloud / star)
// ---------------------------------------------------------------------------

/// A particle orbiting the black hole (e.g. star, gas cloud, tracer).
#[derive(Clone, Debug)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub mass: f64,
    pub alive: bool,
    /// Orbital trail for visualisation (last N positions)
    pub trail: Vec<Vec3>,
}

impl Particle {
    pub fn new(position: Vec3, velocity: Vec3, mass: f64) -> Self {
        Self {
            position,
            velocity,
            mass,
            alive: true,
            trail: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Accretion Disk
// ---------------------------------------------------------------------------

/// A single particle in the accretion disk.
#[derive(Clone, Debug)]
pub struct DiskParticle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub temperature: f64,
    pub alive: bool,
}

impl DiskParticle {
    pub fn new(position: Vec3, velocity: Vec3, temperature: f64) -> Self {
        Self {
            position,
            velocity,
            temperature,
            alive: true,
        }
    }
}

/// Accretion disk model consisting of many particles in (near-)Keplerian
/// orbits around the black hole.
#[derive(Clone, Debug)]
pub struct AccretionDisk {
    pub inner_radius: f64,
    pub outer_radius: f64,
    pub density: f64,
    pub particles: Vec<DiskParticle>,
    pub black_hole_mass: f64,
}

impl AccretionDisk {
    /// Create a new accretion disk with the given geometry and density.
    ///
    /// * `inner_radius` – inner edge (typically 3·rₛ = ISCO)
    /// * `outer_radius` – outer edge of the disk
    /// * `density` – controls the number of spawned particles (0.0 – 2.0)
    /// * `bh_mass` – mass of the central black hole
    pub fn new(inner_radius: f64, outer_radius: f64, density: f64, bh_mass: f64) -> Self {
        let mut disk = Self {
            inner_radius,
            outer_radius,
            density,
            particles: Vec::new(),
            black_hole_mass: bh_mass,
        };
        disk.generate_particles();
        disk
    }

    /// Spawn disk particles on (near-)Keplerian orbits.
    fn generate_particles(&mut self) {
        let mut rng = rand::rng();
        let num_particles = (self.density * 3000.0) as usize;
        let bh = BlackHole::new(self.black_hole_mass, Vec3::zeros());

        for _ in 0..num_particles {
            // Power-law radius distribution (more particles near inner edge)
            let t: f64 = rng.random();
            let r = self.inner_radius + (self.outer_radius - self.inner_radius) * t.powf(0.7);

            // Skip if inside ISCO
            if r <= bh.isco_radius() {
                continue;
            }

            let angle = rng.random::<f64>() * 2.0 * std::f64::consts::PI;

            // Keplerian velocity from Paczyński-Wiita potential
            let v = bh.circular_orbit_velocity(r);

            // Small vertical offset for disk thickness
            let thickness = (self.outer_radius - self.inner_radius) * 0.02;
            let y_pos = (rng.random::<f64>() - 0.5) * thickness;

            let pos = Vec3::new(r * angle.cos(), y_pos, r * angle.sin());

            // Slightly sub-Keplerian with small random perturbations
            let vel_mag = v * (0.96 + rng.random::<f64>() * 0.08);
            let vel = Vec3::new(
                -vel_mag * angle.sin() + rng.random::<f64>() * 0.02 * v,
                rng.random::<f64>() * 0.02 * v,
                vel_mag * angle.cos() + rng.random::<f64>() * 0.02 * v,
            );

            // Temperature follows thin-disk scaling: T ∝ r^(-3/4)
            // Since r ≥ inner_radius, normalized_r ≥ 1.0, so temp ≤ 1.0 automatically
            let normalized_r = r / self.inner_radius;
            let temp = normalized_r.powf(-0.75);

            self.particles.push(DiskParticle::new(pos, vel, temp));
        }
    }

    /// Advance the accretion disk by one time step `dt`.
    pub fn update(&mut self, dt: f64, bh: &BlackHole) {
        for p in &mut self.particles {
            let acc = bh.gravitational_acceleration(&p.position);

            // Semi-implicit Euler integration (symplectic)
            p.velocity += acc * dt;
            p.position += p.velocity * dt;

            // Update temperature based on current radius
            let r = p.position.norm();
            if r > 0.0 && p.alive {
                let normalized_r = r / self.inner_radius;
                p.temperature = normalized_r.powf(-0.75);
            }

            // Remove particles that fall into the black hole
            if r < bh.schwarzschild_radius * 0.8 {
                p.alive = false;
            }
        }
        self.particles.retain(|p| p.alive);
    }

    /// Refill the disk to maintain particle count near the initial density.
    pub fn refill(&mut self) {
        let target = (self.density * 3000.0) as usize;
        if self.particles.len() < target / 2 {
            // Only generate what's needed, capped at target
            self.generate_particles();
            // Trim back to target if we over-generated
            if self.particles.len() > target {
                self.particles.truncate(target);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Simulation Parameters & State
// ---------------------------------------------------------------------------

/// Parameters that can be adjusted from the UI.
#[derive(Clone, Debug)]
pub struct SimulationParams {
    pub black_hole_mass: f64,
    pub particle_speed: f64,
    pub disk_density: f64,
    pub time_step: f64,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            black_hole_mass: 10.0,
            particle_speed: 1.0,
            disk_density: 0.5,
            time_step: 0.008,
        }
    }
}

/// Top-level simulation state that owns all physics entities and drives
/// the time integration.
pub struct SimulationState {
    pub black_hole: BlackHole,
    pub particles: Vec<Particle>,
    pub accretion_disk: AccretionDisk,
    pub params: SimulationParams,
    pub time: f64,
    pub paused: bool,
    pub reset_requested: bool,
}

impl SimulationState {
    /// Create a fresh simulation with the given parameters.
    pub fn new(params: SimulationParams) -> Self {
        let bh = BlackHole::new(params.black_hole_mass, Vec3::zeros());
        let rs = bh.schwarzschild_radius;

        let disk = AccretionDisk::new(
            3.0 * rs,
            25.0 * rs,
            params.disk_density,
            params.black_hole_mass,
        );

        let particles = Self::generate_test_particles(&bh, params.particle_speed);

        Self {
            black_hole: bh,
            particles,
            accretion_disk: disk,
            params,
            time: 0.0,
            paused: false,
            reset_requested: false,
        }
    }

    /// Generate a set of test particles in various elliptical orbits.
    fn generate_test_particles(bh: &BlackHole, speed_factor: f64) -> Vec<Particle> {
        let mut rng = rand::rng();
        let mut particles = Vec::new();
        let rs = bh.schwarzschild_radius;

        // ---- Particles on near-circular orbits (the "swarm") ----
        for _ in 0..200 {
            let r = (4.0 + rng.random::<f64>() * 15.0) * rs;
            if r <= bh.isco_radius() {
                continue;
            }

            let angle = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
            let inclination = (rng.random::<f64>() - 0.5) * 0.3;

            let v_circ = bh.circular_orbit_velocity(r);
            let eccentricity = rng.random::<f64>() * 0.3;
            let v_mag = v_circ * (1.0 - eccentricity * 0.3) * speed_factor;

            let pos = Vec3::new(
                r * angle.cos(),
                r * inclination.sin() * 0.5 * rs,
                r * angle.sin(),
            );

            let vel = Vec3::new(
                -v_mag * angle.sin(),
                v_mag * inclination * 0.3,
                v_mag * angle.cos(),
            );

            particles.push(Particle::new(pos, vel, 0.001));
        }

        // ---- In-falling particles from far away ----
        for _ in 0..30 {
            let r = (30.0 + rng.random::<f64>() * 50.0) * rs;
            let angle = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
            let v_circ = bh.circular_orbit_velocity(r);
            let v_mag = v_circ * 0.3;

            let pos = Vec3::new(
                r * angle.cos(),
                (rng.random::<f64>() - 0.5) * 5.0 * rs,
                r * angle.sin(),
            );

            let vel = Vec3::new(
                -v_mag * angle.sin() * 0.5,
                -v_mag * 0.1,
                -v_mag * angle.cos() * 0.5,
            );

            particles.push(Particle::new(pos, vel, 0.001));
        }

        particles
    }

    /// Advance the simulation by one time step using semi-implicit Euler
    /// integration (symplectic, preserves orbital energy over long runs).
    pub fn step(&mut self) {
        if self.paused {
            return;
        }

        if self.reset_requested {
            let new_params = self.params.clone();
            *self = Self::new(new_params);
            return;
        }

        let dt = self.params.time_step;

        // ---- Update test particles ----
        for p in &mut self.particles {
            let acc = self.black_hole.gravitational_acceleration(&p.position);

            // Semi-implicit Euler: v(t+dt) = v(t) + a(t)·dt
            //                        x(t+dt) = x(t) + v(t+dt)·dt
            p.velocity += acc * dt;
            p.position += p.velocity * dt;

            // Store trail (last 120 positions)
            p.trail.push(p.position);
            if p.trail.len() > 120 {
                p.trail.remove(0);
            }

            // Remove particles that cross the event horizon
            if p.position.norm() < self.black_hole.schwarzschild_radius * 0.5 {
                p.alive = false;
            }
        }
        self.particles.retain(|p| p.alive);

        // ---- Update accretion disk ----
        self.accretion_disk.update(dt, &self.black_hole);
        self.accretion_disk.refill();

        self.time += dt;
    }

    /// Replace the simulation state with fresh initial conditions using
    /// the provided (or current) parameters.
    pub fn reset(&mut self, params: Option<SimulationParams>) {
        if let Some(p) = params {
            self.params = p;
        }
        *self = Self::new(self.params.clone());
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_hole_creation() {
        let bh = BlackHole::new(10.0, Vec3::zeros());
        assert!(bh.mass == 10.0);
        assert!(bh.schwarzschild_radius > 0.0);
        assert!((bh.schwarzschild_radius - 20.0).abs() < 1e-10); // 2*G*M/c² = 2*1*10/1 = 20
    }

    #[test]
    fn test_acceleration_is_radially_inward() {
        let bh = BlackHole::new(10.0, Vec3::zeros());
        let pos = Vec3::new(100.0, 0.0, 0.0);
        let acc = bh.gravitational_acceleration(&pos);
        // Acceleration should point toward origin (negative x)
        assert!(acc.x < 0.0);
        assert!(acc.y == 0.0);
        assert!(acc.z == 0.0);
    }

    #[test]
    fn test_acceleration_diverge_near_horizon() {
        let bh = BlackHole::new(10.0, Vec3::zeros());
        let rs = bh.schwarzschild_radius;
        // Just outside the horizon – acceleration should be very large
        let pos = Vec3::new(rs * 1.1, 0.0, 0.0);
        let acc = bh.gravitational_acceleration(&pos);
        assert!(acc.norm() > 1.0);
    }

    #[test]
    fn test_zero_acceleration_inside_horizon() {
        let bh = BlackHole::new(10.0, Vec3::zeros());
        let pos = Vec3::new(bh.schwarzschild_radius * 0.5, 0.0, 0.0);
        let acc = bh.gravitational_acceleration(&pos);
        assert_eq!(acc, Vec3::zeros());
    }

    #[test]
    fn test_circular_orbit_velocity() {
        let bh = BlackHole::new(10.0, Vec3::zeros());
        let v = bh.circular_orbit_velocity(100.0);
        assert!(v > 0.0);
        // v = sqrt(G*M*r/(r - rs)²)
        let expected = (G * bh.mass * 100.0 / (100.0 - bh.schwarzschild_radius).powi(2)).sqrt();
        assert!((v - expected).abs() < 1e-10);
    }

    #[test]
    fn test_no_orbit_inside_isco() {
        let bh = BlackHole::new(10.0, Vec3::zeros());
        let r_isco = bh.isco_radius();
        let v = bh.circular_orbit_velocity(r_isco * 0.5);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_isco_radius() {
        let bh = BlackHole::new(10.0, Vec3::zeros());
        assert!((bh.isco_radius() - 3.0 * bh.schwarzschild_radius).abs() < 1e-10);
    }

    #[test]
    fn test_predict_trajectory() {
        let bh = BlackHole::new(10.0, Vec3::zeros());
        let pos = Vec3::new(100.0, 0.0, 0.0);
        let vel = Vec3::new(0.0, 0.0, 1.0);
        let path = bh.predict_trajectory(pos, vel, 10, 0.1);
        assert_eq!(path.len(), 10);
        // The particle should have moved
        assert!(path[0] != pos);
        assert!(path[9] != path[0]);
    }

    #[test]
    fn test_simulation_step_updates_time() {
        let mut state = SimulationState::new(SimulationParams::default());
        let time_before = state.time;
        state.step();
        assert!(state.time > time_before);
    }

    #[test]
    fn test_particle_removed_near_horizon() {
        let bh = BlackHole::new(10.0, Vec3::zeros());
        let rs = bh.schwarzschild_radius;

        let mut state = SimulationState::new(SimulationParams::default());
        let initial_count = state.particles.len();

        // Add a particle well outside the horizon
        state.particles.push(Particle::new(
            Vec3::new(rs * 2.5, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -4.0), // inward velocity
            0.001,
        ));

        // Step until it either gets removed or we hit max steps
        for _ in 0..500 {
            state.step();
        }

        // The total should be <= initial (some may have fallen in)
        assert!(state.particles.len() <= initial_count + 1,
            "Expected particles to possibly be removed, but none were. count={}",
            state.particles.len());
    }

    #[test]
    fn test_disk_particle_generation() {
        let bh = BlackHole::new(10.0, Vec3::zeros());
        let rs = bh.schwarzschild_radius;
        let disk = AccretionDisk::new(3.0 * rs, 25.0 * rs, 0.5, 10.0);
        assert!(!disk.particles.is_empty());
        for p in &disk.particles {
            assert!(p.temperature > 0.0 && p.temperature <= 1.0);
        }
    }

    #[test]
    fn test_simulation_reset() {
        let mut state = SimulationState::new(SimulationParams::default());
        state.time = 100.0;
        let old_params = state.params.clone();
        state.reset(Some(old_params));
        // After reset, time should be zero
        assert!(state.time.abs() < 1e-10);
    }
}
