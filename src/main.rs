mod physics;
mod render;

use std::sync::Arc;
use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;
use render::{Camera, RenderResources};

struct BlackHoleApp {
    sim: physics::SimulationState,
    camera: Camera,
}

struct SimulationCallback;

impl egui_wgpu::CallbackTrait for SimulationCallback {
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(resources) = callback_resources.get::<RenderResources>() {
            render_pass.set_pipeline(&resources.pipeline);
            render_pass.set_bind_group(0, &resources.bind_group, &[]);
            render_pass.draw(0..3, 0..1); // Draw the full screen triangle
        }
    }
}

impl BlackHoleApp {
    fn new() -> Self {
        let params = physics::SimulationParams::default();
        let sim = physics::SimulationState::new(params);
        Self {
            sim,
            camera: Camera::new(),
        }
    }
}

impl eframe::App for BlackHoleApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        // Run simulation step if not paused
        if !self.sim.paused {
            self.sim.step();
            // Force redraw every frame for animation
            ui.ctx().request_repaint();
        }

        // Draw side panel for UI controls
        egui::Panel::right("controls_panel")
            .resizable(false)
            .default_size(260.0)
            .show(ui, |ui: &mut egui::Ui| {
                ui.vertical_centered(|ui: &mut egui::Ui| {
                    ui.add_space(10.0);
                    ui.heading("🌀 BLACK HOLE");
                    ui.label("Simulator Control Panel");
                    ui.add_space(15.0);
                });

                ui.separator();
                ui.add_space(10.0);

                ui.group(|ui: &mut egui::Ui| {
                    ui.label("Simulation Settings");
                    ui.add_space(5.0);

                    // Black Hole Mass
                    ui.label(format!("Black Hole Mass: {:.1}", self.sim.params.black_hole_mass));
                    let old_mass = self.sim.params.black_hole_mass;
                    ui.add(egui::Slider::new(&mut self.sim.params.black_hole_mass, 1.0..=30.0).show_value(false));
                    if self.sim.params.black_hole_mass != old_mass {
                        // Mass change triggers reset of the horizon and inner orbits
                        self.sim.reset(None);
                    }

                    ui.add_space(8.0);

                    // Particle Speed (acts as step speed/fidelity)
                    ui.label(format!("Fidelity: {:.2}x", self.sim.params.particle_speed));
                    let old_speed = self.sim.params.particle_speed;
                    ui.add(egui::Slider::new(&mut self.sim.params.particle_speed, 0.1..=3.0).show_value(false));
                    if self.sim.params.particle_speed != old_speed {
                        self.sim.reset(None);
                    }

                    ui.add_space(8.0);

                    // Accretion Disk Density (mapped to brightness)
                    ui.label(format!("Disk Density: {:.2}", self.sim.params.disk_density));
                    let old_density = self.sim.params.disk_density;
                    ui.add(egui::Slider::new(&mut self.sim.params.disk_density, 0.1..=2.0).show_value(false));
                    if self.sim.params.disk_density != old_density {
                        self.sim.reset(None);
                    }
                });

                ui.add_space(15.0);

                ui.group(|ui: &mut egui::Ui| {
                    ui.label("Simulation Controls");
                    ui.add_space(5.0);

                    ui.horizontal(|ui: &mut egui::Ui| {
                        if ui.button(if self.sim.paused { "▶ Play" } else { "⏸ Pause" }).clicked() {
                            self.sim.paused = !self.sim.paused;
                            if !self.sim.paused {
                                ui.ctx().request_repaint();
                            }
                        }

                        if ui.button("🔄 Reset").clicked() {
                            self.sim.reset(None);
                            ui.ctx().request_repaint();
                        }
                    });
                });

                ui.add_space(15.0);

                ui.group(|ui: &mut egui::Ui| {
                    ui.label("Simulation Info");
                    ui.add_space(5.0);
                    
                    let fps = 1.0 / ui.ctx().input(|i| i.stable_dt);
                    ui.label(format!("FPS: {:.1}", fps));
                    
                    ui.label(format!("Time: {:.3}", self.sim.time));
                    ui.label(format!("Schwarzschild Radius: {:.2}", self.sim.black_hole.schwarzschild_radius));
                    ui.label(format!("ISCO Radius: {:.2}", self.sim.black_hole.isco_radius()));
                });

                ui.add_space(15.0);
                ui.vertical_centered(|ui: &mut egui::Ui| {
                    ui.weak("Drag mouse on view to rotate");
                    ui.weak("Scroll mouse wheel to zoom");
                });
            });

        // Draw Central Panel containing the 3D visualization
        egui::CentralPanel::default().show(ui, |ui: &mut egui::Ui| {
            let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

            // Handle Camera Rotation via dragging
            if response.dragged_by(egui::PointerButton::Primary) {
                let drag = response.drag_delta();
                self.camera.theta -= drag.x * 0.007;
                self.camera.phi = (self.camera.phi - drag.y * 0.007).clamp(0.1, std::f32::consts::PI - 0.1);
                ui.ctx().request_repaint();
            }

            // Handle Camera Zoom via scroll
            let scroll = ui.input(|i| i.smooth_scroll_delta);
            if scroll.y != 0.0 {
                self.camera.radius = (self.camera.radius - scroll.y * 0.2).clamp(10.0, 500.0);
                ui.ctx().request_repaint();
            }

            // Render wgpu callback
            if let Some(render_state) = frame.wgpu_render_state() {
                // Update uniforms on the GPU
                let mut renderer = render_state.renderer.write();
                let callback_resources = &mut renderer.callback_resources;
                if let Some(resources) = callback_resources.get_mut::<RenderResources>() {
                    resources.update_uniforms(
                        &render_state.queue,
                        &self.camera,
                        &self.sim,
                        rect.width(),
                        rect.height(),
                    );
                }

                // Register rendering callback using egui_wgpu::Callback
                let paint_callback = egui_wgpu::Callback::new_paint_callback(
                    rect,
                    SimulationCallback,
                );

                ui.painter().add(paint_callback);
            }
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("🌀 Black Hole Simulator")
            .with_inner_size([1100.0, 750.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Black Hole Simulator",
        options,
        Box::new(|cc| {
            if let Some(render_state) = &cc.wgpu_render_state {
                let resources = RenderResources::new(&render_state.device, render_state.target_format);
                let mut renderer = render_state.renderer.write();
                renderer.callback_resources.insert(resources);
            }
            Ok(Box::new(BlackHoleApp::new()))
        }),
    )
}
