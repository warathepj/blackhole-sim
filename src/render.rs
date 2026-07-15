use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;
use nalgebra::{Matrix4, Point3, Vector3 as NVec3};

use crate::physics::SimulationState;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub size: f32,
}

impl ParticleVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ParticleVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 4]>() ) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

pub struct Camera {
    pub theta: f32, // Azimuthal angle
    pub phi: f32,   // Polar angle
    pub radius: f32, // Distance from center
    pub target: Point3<f32>,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            theta: 0.8,
            phi: 1.2,
            radius: 120.0,
            target: Point3::origin(),
        }
    }

    pub fn to_view_proj(&self, aspect_ratio: f32) -> Matrix4<f32> {
        let x = self.radius * self.phi.sin() * self.theta.cos();
        let y = self.radius * self.phi.cos();
        let z = self.radius * self.phi.sin() * self.theta.sin();

        let eye = Point3::new(x, y, z);
        let view = Matrix4::look_at_rh(&eye, &self.target, &NVec3::new(0.0, 1.0, 0.0));
        let proj = Matrix4::new_perspective(aspect_ratio, 45.0f32.to_radians(), 0.1, 1000.0);

        proj * view
    }
}

pub struct RenderResources {
    pub pipeline: wgpu::RenderPipeline,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub vertex_count: u32,
}

impl RenderResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Simulation Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        // Camera Uniform Buffer
        let camera_uniform = CameraUniform {
            view_proj: Matrix4::identity().into(),
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[Some(&camera_bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Simulation Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[ParticleVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            camera_buffer,
            camera_bind_group,
            vertex_buffer: None,
            vertex_count: 0,
        }
    }

    pub fn update_particles(&mut self, device: &wgpu::Device, _queue: &wgpu::Queue, sim: &SimulationState) {
        let mut vertices = Vec::new();

        // 1. Render Black Hole event horizon as a dense point sphere/shell for visualization
        let bh = &sim.black_hole;
        let rs = bh.schwarzschild_radius;
        // Generate points on sphere
        let num_bh_points = 500;
        for i in 0..num_bh_points {
            let t = i as f32 / num_bh_points as f32;
            let phi = t * std::f32::consts::PI;
            let theta = t * 2.0 * std::f32::consts::PI * 13.0; // spiral-like
            
            let x = rs as f32 * phi.sin() * theta.cos();
            let y = rs as f32 * phi.cos();
            let z = rs as f32 * phi.sin() * theta.sin();

            vertices.push(ParticleVertex {
                position: [x, y, z],
                color: [0.03, 0.03, 0.03, 1.0], // Jet black horizon
                size: 4.0,
            });
        }

        // 2. Render Accretion Disk Particles
        for p in &sim.accretion_disk.particles {
            let t = p.temperature as f32;
            let color = if t > 0.8 {
                [1.0, 0.95, 0.9, 1.0] // bright yellowish white
            } else if t > 0.4 {
                let factor = (t - 0.4) / 0.4;
                [1.0, 0.6 + 0.35 * factor, 0.1 * factor, 1.0] // Orange to yellow
            } else {
                let factor = t / 0.4;
                [0.4 + 0.6 * factor, 0.05 + 0.25 * factor, 0.02, 1.0] // Dark red to orange
            };

            vertices.push(ParticleVertex {
                position: [p.position.x as f32, p.position.y as f32, p.position.z as f32],
                color,
                size: 2.0,
            });
        }

        // 3. Render Test Particles
        for p in &sim.particles {
            vertices.push(ParticleVertex {
                position: [p.position.x as f32, p.position.y as f32, p.position.z as f32],
                color: [0.4, 0.7, 1.0, 0.9], // light blue
                size: 3.0,
            });

            // Also render trails
            for (idx, trail_pos) in p.trail.iter().enumerate() {
                let alpha = (idx as f32 / p.trail.len() as f32) * 0.3;
                vertices.push(ParticleVertex {
                    position: [trail_pos.x as f32, trail_pos.y as f32, trail_pos.z as f32],
                    color: [0.4, 0.7, 1.0, alpha],
                    size: 1.0,
                });
            }
        }

        self.vertex_count = vertices.len() as u32;

        if self.vertex_count > 0 {
            self.vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Simulation Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }));
        } else {
            self.vertex_buffer = None;
        }
    }
}
