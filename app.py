import panel as pn
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import numpy as np
from branca.colormap import LinearColormap
from datetime import datetime
import json

pn.extension('tabulator', sizing_mode='stretch_width')

class RealEstateDashboard:
    def __init__(self):
        # Configuration
        self.map_size = (1200, 800)
        self.kenya_center = [0.0236, 37.9062]
        self.default_zoom = 7

        # Load and process data
        self.load_data()
        self.process_data()

        # Create widgets
        self.create_widgets()

        # Create initial map and layout
        self.map_pane = pn.pane.HTML(self.create_map()._repr_html_(), 
                                    width=self.map_size[0], 
                                    height=self.map_size[1])
        self.layout = self.create_layout()

    def serialize_dates(self, df):
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].apply(
                    lambda x: x.isoformat() if pd.notnull(x) else "Unknown"
                )
        return df

    def clean_geodata(self, gdf):
        gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
        gdf = self.serialize_dates(gdf)
        for col in gdf.columns:
            if gdf[col].isna().any():
                if gdf[col].dtype == 'object':
                    gdf[col] = gdf[col].fillna('Unknown')
                else:
                    gdf[col] = gdf[col].fillna(0)
        return gdf

    def load_data(self):
        try:
            self.full_grid = self.clean_geodata(
                gpd.read_file("data/full_grid.geojson").to_crs(epsg=4326)
            )
            self.roads = self.clean_geodata(
                gpd.read_file("data/roads.geojson").to_crs(epsg=4326)
            )
            self.towns = self.clean_geodata(
                gpd.read_file("data/towns.geojson").to_crs(epsg=4326)
            )
            self.flood = self.clean_geodata(
                gpd.read_file("data/flood_areas.geojson").to_crs(epsg=4326)
            )
            self.wards = self.clean_geodata(
                gpd.read_file("Kenya_Wards/kenya_wards.shp").to_crs(epsg=4326)
            )

            if not self.flood.empty:
                if 'Area_ha' not in self.flood.columns:
                    if 'SHAPE_Area' in self.flood.columns:
                        self.flood['Area_ha'] = self.flood['SHAPE_Area'] / 10000
                    else:
                        self.flood['Area_ha'] = self.flood.geometry.area / 10000
                self.flood = self.flood[self.flood['Area_ha'] >= 0.1]
                self.flood.geometry = self.flood.geometry.simplify(0.001)

            name_cols = [c for c in ['name', 'town_name', 'NAME', 'TOWN_NAME'] if c in self.towns.columns]
            self.towns['name'] = self.towns[name_cols[0]] if name_cols else 'Town ' + self.towns.index.astype(str)

            if 'road_type' not in self.roads.columns:
                self.roads['road_type'] = 'road'

        except Exception as e:
            print(f"Data loading error: {str(e)}")
            raise

    def process_data(self):
        components = ['infra_score', 'env_score', 'suitability', 'mean_sentiment']
        for col in components:
            if col in self.full_grid.columns:
                col_min = self.full_grid[col].min()
                col_range = self.full_grid[col].max() - col_min
                self.full_grid[col] = (
                    (self.full_grid[col] - col_min) / col_range 
                    if col_range > 0 else 0.5
                ).clip(0, 1)
            else:
                self.full_grid[col] = 0.5

        weights = {
            'infra_score': 0.4,
            'env_score': 0.3,
            'suitability': 0.2,
            'mean_sentiment': 0.1
        }
        self.full_grid['opportunity_score'] = sum(
            self.full_grid[col] * weight 
            for col, weight in weights.items()
        )

        try:
            self.full_grid['opportunity_class'] = pd.qcut(
                self.full_grid['opportunity_score'],
                q=[0, 0.2, 0.8, 1],
                labels=['Low', 'Medium', 'High']
            )
        except Exception as e:
            print(f"Classification error: {str(e)}")
            self.full_grid['opportunity_class'] = 'Medium'

    def create_widgets(self):
        self.opportunity_filter = pn.widgets.Select(
            name='Opportunity Class', 
            options=['All', 'Low', 'Medium', 'High'], 
            value='All'
        )
        self.heatmap_radius = pn.widgets.IntSlider(
            name='Heatmap Radius', start=10, end=50, value=25, step=5)
        self.heatmap_intensity = pn.widgets.FloatSlider(
            name='Heatmap Intensity', start=0.1, end=1.0, value=0.7, step=0.1)
        self.flood_color = pn.widgets.ColorPicker(name='Flood Zone Color', value='#1E90FF')
        self.flood_opacity = pn.widgets.FloatSlider(name='Flood Zone Opacity', start=0.3, end=1.0, value=0.6, step=0.1)
        self.flood_edge_color = pn.widgets.ColorPicker(name='Flood Zone Edge Color', value='#0000FF')
        self.flood_edge_width = pn.widgets.IntSlider(name='Flood Zone Edge Width', start=1, end=5, value=2, step=1)
        self.show_roads = pn.widgets.Checkbox(name='Show Roads', value=True)
        self.show_towns = pn.widgets.Checkbox(name='Show Towns', value=True)
        self.show_flood = pn.widgets.Checkbox(name='Show Flood Zones', value=True)

        for widget in [self.opportunity_filter, self.heatmap_radius, self.heatmap_intensity, self.flood_color,
                      self.flood_opacity, self.flood_edge_color, self.flood_edge_width, self.show_roads,
                      self.show_towns, self.show_flood]:
            widget.param.watch(self.update_map, 'value')

    def create_map(self):
        m = folium.Map(
            location=self.kenya_center,
            zoom_start=self.default_zoom,
            tiles='CartoDB positron',
            width='100%',
            height='100%'
        )

        filtered_grid = (
            self.full_grid if self.opportunity_filter.value == 'All'
            else self.full_grid[self.full_grid['opportunity_class'] == self.opportunity_filter.value]
        )

        if self.show_flood.value and 'flooded' in self.full_grid.columns:
            flooded_only = self.full_grid[self.full_grid['flooded'] == 1]
            for _, row in flooded_only.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=4,
                    color=self.flood_edge_color.value,
                    fill=True,
                    fill_color=self.flood_color.value,
                    fill_opacity=self.flood_opacity.value,
                    weight=self.flood_edge_width.value,
                    popup=folium.Popup("Flooded Point", max_width=150)
                ).add_to(m)

        if self.show_roads.value and not self.roads.empty:
            folium.GeoJson(
                self.roads.__geo_interface__,
                style_function=lambda x: {
                    'color': '#636363',
                    'weight': 2.5,
                    'opacity': 0.7
                },
                name='Roads'
            ).add_to(m)

        if self.show_towns.value and not self.towns.empty:
            town_cluster = MarkerCluster(name="Towns").add_to(m)
            for _, row in self.towns.iterrows():
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=f"<b>{row['name']}</b>",
                    icon=folium.Icon(color='red', icon='home', prefix='fa')
                ).add_to(town_cluster)

        if not filtered_grid.empty:
            HeatMap(
                [[point.y, point.x, score * self.heatmap_intensity.value] 
                 for point, score in zip(filtered_grid.geometry.centroid, filtered_grid['opportunity_score'])],
                name='Opportunity Heatmap',
                radius=self.heatmap_radius.value,
                blur=20,
                gradient={0.0: 'blue', 0.3: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'},
                min_opacity=0.4
            ).add_to(m)

            colormap = LinearColormap(
                ['blue', 'cyan', 'lime', 'yellow', 'red'],
                vmin=0,
                vmax=1,
                caption='Opportunity Score'
            )
            colormap.add_to(m)

        if hasattr(self, 'wards') and not self.wards.empty:
            folium.GeoJson(
                self.wards.__geo_interface__,
                style_function=lambda x: {'color': '#999999', 'weight': 0.5, 'fillOpacity': 0},
                tooltip=folium.GeoJsonTooltip(
                    fields=['ward', 'subcounty', 'county'],
                    aliases=['Ward:', 'Subcounty:', 'County:'],
                    sticky=True
                ),
                name='Wards'
            ).add_to(m)

        folium.LayerControl(collapsed=False, position='topright').add_to(m)
        return m

    def update_map(self, event):
        self.map_pane.object = self.create_map()._repr_html_()

    def create_layout(self):
        flood_controls = pn.Card(
            pn.pane.Markdown("### Flood Zone Styling"),
            pn.Row(self.flood_color, self.flood_edge_color),
            pn.Row(self.flood_opacity, self.flood_edge_width),
            title="Flood Zone Settings",
            collapsed=False,
            styles={'background': '#f0f8ff'}
        )

        heatmap_controls = pn.Card(
            pn.pane.Markdown("### Heatmap Settings"),
            self.heatmap_radius,
            self.heatmap_intensity,
            title="Heatmap Controls",
            collapsed=False
        )

        layer_controls = pn.Card(
            pn.pane.Markdown("### Layer Visibility"),
            self.opportunity_filter,
            self.show_roads,
            self.show_towns,
            self.show_flood,
            title="Map Layers",
            collapsed=False
        )

        controls = pn.Column(
            flood_controls,
            heatmap_controls,
            layer_controls,
            width=350,
            sizing_mode='fixed'
        )

        return pn.Row(
            controls,
            self.map_pane,
            sizing_mode='stretch_width',
            margin=(10, 10, 10, 10)
        )

    def serve(self):
        return self.layout.servable()

# Create and serve dashboard
dashboard = RealEstateDashboard()
dashboard.serve()