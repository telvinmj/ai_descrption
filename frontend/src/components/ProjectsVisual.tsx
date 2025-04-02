/**
 * DEPRECATED: This component is no longer used and should be removed.
 * It was previously used to display a visualization of projects.
 */

import React from 'react';

interface Project {
  id: string;
  name: string;
  description: string;
}

interface Model {
  id: string;
  project: string;
}

interface ProjectsVisualProps {
  projects: Project[];
  models: Model[];
}

const ProjectsVisual: React.FC<ProjectsVisualProps> = ({ projects, models }) => {
  // Create a simple visualization of projects and their model counts
  return (
    <div className="projects-visual">
      <div className="projects-grid">
        {projects.map(project => {
          const modelCount = models.filter(model => model.project === project.name).length;
          const percentage = models.length > 0 ? (modelCount / models.length) * 100 : 0;
          
          return (
            <div key={project.id} className="project-box">
              <h3>{project.name}</h3>
              <div className="model-count">{modelCount} models</div>
              <div className="percentage-bar">
                <div 
                  className="fill" 
                  style={{ 
                    width: `${percentage}%`,
                    backgroundColor: `hsl(${Math.floor(Math.random() * 360)}, 70%, 80%)` 
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ProjectsVisual; 