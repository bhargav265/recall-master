import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';
// import * as sqs from 'aws-cdk-lib/aws-sqs';

export class RecallmasterCdkStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, {
      env: {
        account: "977099028034", // Add your AWS account ID
        region: "us-east-1",      // Add your desired AWS region
      },
      ...props,
    });

    // The code that defines your stack goes here

    // example resource
    // const queue = new sqs.Queue(this, 'RecallmasterCdkQueue', {
    //   visibilityTimeout: cdk.Duration.seconds(300)
    // });
    // Create DynamoDB table
    const table = new dynamodb.Table(this, 'RecallmasterTable', {
      partitionKey: { name: 'id', type: dynamodb.AttributeType.STRING },
    });

    // Create Lambda function
    const lambdaFunction = new lambda.DockerImageFunction(this, 'RecallmasterFunction2', {
      functionName: 'RecallmasterFunction2',
      code: lambda.DockerImageCode.fromImageAsset('../lambda'),
      architecture: lambda.Architecture.X86_64,
      environment: {
        DYNAMODB_TABLE_NAME: table.tableName,
      },
      timeout: cdk.Duration.minutes(3),
    });
    // Grant Lambda function read/write permissions to DynamoDB table
    table.grantReadWriteData(lambdaFunction);
    // Create API Gateway
    const api = new apigateway.LambdaRestApi(this, 'RecallmasterApi', {
      handler: lambdaFunction,
      proxy: false,
    });

    const twilioResource = api.root.addResource('twilio');
    const incomingMessageResource = twilioResource.addResource('incoming_message');
    incomingMessageResource.addMethod('POST');
  }
}
