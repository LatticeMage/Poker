FROM node:24

WORKDIR /app

# Run as the unprivileged `node` user (uid 1000) so files written into the
# bind-mounted source tree (.svelte-kit, build) are not owned by root.
RUN chown node:node /app
USER node

# Install dependencies
COPY --chown=node:node package*.json .npmrc ./
RUN npm install

# Copy all files to the container
COPY --chown=node:node . .

RUN npm run build

# Start the application
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "3000"]
