import React from 'react';

interface PageHeaderProps {
  kicker: string;
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
}

export const PageHeader: React.FC<PageHeaderProps> = ({ kicker, title, description, actions }) => {
  return (
    <header className="page-header">
      <div className="page-header-left">
        <div className="hero-kicker">{kicker}</div>
        <h1 className="page-title">{title}</h1>
        {description ? <p className="page-sub">{description}</p> : null}
      </div>
      {actions ? <div className="page-header-actions">{actions}</div> : null}
    </header>
  );
};

export const PlaceholderBadge: React.FC<{ label?: string }> = ({ label = 'Static scaffold' }) => (
  <span className="placeholder-badge" title="UI scaffold; backend not wired yet">
    {label}
  </span>
);
