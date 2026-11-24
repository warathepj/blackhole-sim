struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct LensingUniforms {
    camera_pos: vec3<f32>,
    bh_mass: f32,
    camera_forward: vec3<f32>,
    rs: f32,
    camera_right: vec3<f32>,
    time: f32,
    camera_up: vec3<f32>,
    screen_width: f32,
    screen_height: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: LensingUniforms;

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate a single large triangle that covers the entire viewport
    let x = -1.0 + f32(in_vertex_index == 1u) * 4.0;
    let y = -1.0 + f32(in_vertex_index == 2u) * 4.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

// Helper to cross product for vec3
fn my_cross(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Procedural starfield
fn sample_starfield(dir: vec3<f32>) -> vec4<f32> {
    // Generate a pseudo-random value based on direction
    let theta = atan2(dir.z, dir.x);
    let phi = acos(dir.y);
    
    let scale = 150.0;
    let grid_x = floor(theta * scale);
    let grid_y = floor(phi * scale);
    
    // Hash function
    let val = fract(sin(grid_x * 12.9898 + grid_y * 78.233) * 43758.5453);
    
    if (val > 0.993) {
        let intensity = fract(val * 100.0);
        // Add subtle star twinkle and coloring
        let star_color = vec3<f32>(
            0.8 + 0.2 * sin(val * 10.0),
            0.85 + 0.15 * cos(val * 20.0),
            1.0
        );
        return vec4<f32>(star_color * intensity, 1.0);
    }
    
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

// Map temperature [0, 1] to blackbody color gradient (red -> orange -> yellow -> white)
fn blackbody(temp: f32) -> vec3<f32> {
    if (temp > 0.8) {
        return vec3<f32>(1.0, 0.95, 0.9); // bright yellowish white
    } else if (temp > 0.4) {
        let f = (temp - 0.4) / 0.4;
        return vec3<f32>(1.0, 0.6 + 0.35 * f, 0.1 * f); // Orange to yellow
    } else {
        let f = temp / 0.4;
        return vec3<f32>(0.4 + 0.6 * f, 0.05 + 0.25 * f, 0.02); // Dark red to orange
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect = uniforms.screen_width / uniforms.screen_height;
    
    // Compute initial ray direction from screen coordinate
    let u = in.uv.x * aspect;
    let v = in.uv.y;
    let fov_factor = 1.0; // zoom level
    let ray_dir = normalize(uniforms.camera_forward * fov_factor + u * uniforms.camera_right + v * uniforms.camera_up);
    
    var p = uniforms.camera_pos;
    var vel = ray_dir;
    
    var acc_color = vec3<f32>(0.0, 0.0, 0.0);
    var acc_alpha = 0.0;
    
    // Schwarzschild event horizon and accretion disk dimensions
    let rs = uniforms.rs;
    let inner_radius = 3.0 * rs;
    let outer_radius = 18.0 * rs;
    
    // Run numerical integration (raymarching geodesics)
    var step_count = 0;
    let max_steps = 120;
    
    for (var i = 0; i < max_steps; i++) {
        let r = length(p);
        
        // 1. Inside event horizon -> light is trapped
        if (r <= rs * 0.99) {
            acc_color = mix(acc_color, vec3<f32>(0.0), 1.0 - acc_alpha);
            acc_alpha = 1.0;
            break;
        }
        
        // 2. Escaped to infinity -> render background starfield
        if (r > 300.0) {
            break;
        }
        
        // Adaptive step size based on distance to event horizon
        let dt = min(0.4, max(0.02, 0.3 * (r - rs)));
        
        // Save previous position to check plane crossing
        let prev_p = p;
        
        // Relativistic acceleration: a = -1.5 * rs * L^2 / r^5 * p
        let h2 = length(my_cross(p, vel)) * length(my_cross(p, vel));
        let a = -1.5 * rs * h2 / pow(r, 5.0) * p;
        
        // Update ray position & velocity (Euler step)
        vel += a * dt;
        vel = normalize(vel);
        p += vel * dt;
        
        // 3. Check equatorial plane crossing (y = 0) for accretion disk
        if (prev_p.y * p.y < 0.0) {
            // Linear interpolation to find precise crossing point
            let t_cross = -prev_p.y / (p.y - prev_p.y);
            let p_cross = prev_p + vel * (t_cross * dt);
            let r_cross = length(p_cross);
            
            if (r_cross >= inner_radius && r_cross <= outer_radius) {
                // Relativistic Doppler / Redshift / Beaming simplified (cool disk glow)
                let normalized_r = r_cross / inner_radius;
                let temp = pow(normalized_r, -0.75);
                let base_color = blackbody(temp);
                
                // Add noise/dust lanes for disk structure
                let angle = atan2(p_cross.z, p_cross.x);
                let spiral = sin(angle * 3.0 - normalized_r * 4.0 + uniforms.time * 2.0);
                let dust = mix(0.45, 1.0, 0.5 + 0.5 * spiral);
                
                let density = 0.5 * dust * exp(-0.15 * (r_cross - inner_radius));
                let alpha = (1.0 - acc_alpha) * density;
                
                acc_color += base_color * alpha;
                acc_alpha += alpha;
                
                if (acc_alpha >= 0.98) {
                    break;
                }
            }
        }
    }
    
    // Sample background starfield for rays that escaped
    if (acc_alpha < 1.0) {
        let stars = sample_starfield(vel);
        acc_color += stars.rgb * (1.0 - acc_alpha);
    }
    
    return vec4<f32>(acc_color, 1.0);
}
