# Stage 1: Build the React application
FROM node:20.19 AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Stage 2: Serve the built application with Nginx
FROM nginx:alpine

# Copy the build output from the previous stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy your custom nginx configuration if you have one
# COPY ./nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]